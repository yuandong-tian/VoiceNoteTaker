# Transcribe youtube video to text

import argparse
import os
import re
import openai
from subprocess import check_output

openai.api_key = os.environ.get("OPENAI_API_KEY")

file_matcher = re.compile(r"\[ffmpeg\] Correcting container in \"(.*?)\"") 
file_matcher2 = re.compile(r"\[download\] (.*?) has already been downloaded")
file_matcher3 = re.compile(r"\[download\] Destination: (.*?)")

def go_through_matchers(output: str, matchers: list[re.Pattern], match_all=False):
    results = []
    for matcher in matchers:
        m = matcher.match(output)
        if m:
            if match_all:
                results.append(m.group(1).strip())
            else:
                return m.group(1).strip()
    return results

def download_youtube(video_url: str) -> str:
    # First download the video using yt-dlp
    output = check_output(f"yt-dlp --cookies ../youtube_cookie.txt -f 140 {video_url}", shell=True).decode("utf-8")
    file_matchers = [file_matcher, file_matcher2, file_matcher3]
    return go_through_matchers(output, file_matchers)

ffmpeg_matcher = re.compile(r"\[segment @ [^\]]+\] Opening '(.*?)' for writing")

def transcribe_file(audio_file, segment_time=None):
    if segment_time is not None:
        # Segment it
        output = check_output(f"ffmpeg -i {audio_file} -map 0 -f segment -segment_time {segment_time} -reset_timestamps 1 -c copy \"tmp%03d.m4a\"", shell=True)
        audio_files = go_through_matchers(output, [ffmpeg_matcher], match_all=True)
    else:
        audio_files = [audio_file]

    results = ""
    for i, audio_file in enumerate(audio_files):
        with open(audio_file, "rb") as file:   
            response = openai.Audio.transcribe('whisper-1', file)
            if len(audio_files) > 1:
                results += f"Segment {audio_file}: {i} of {len(audio_files)}\n"
            results += response['text'] + "\n"
    return results

if __name__ == "__main__":
    # Use argparse to get the video url
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_url", type=str, default=None)
    parser.add_argument("--audio_files", type=str, default=None, help="comma-separated audio files")
    parser.add_argument("--output", type=str, default="transcribed.txt")
    parser.add_argument("--segment_time", type=int, default=None)
    args = parser.parse_args()

    if args.video_url:
        audio_files = download_youtube(args.video_url)
    else:
        audio_files = args.audio_files.split(",")
    
    results = ""
    for audio_file in audio_files:
        transcribed_text = transcribe_file(audio_file, args.segment_time)
        results += f"\n\nFile {audio_file}\n"
        results += transcribed_text 

    # save the transcribed text to a file
    with open(args.output, "w") as file:
        file.write(results)
