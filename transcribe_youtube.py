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

def download_youtube(video_url: str) -> str:
    # First download the video using yt-dlp
    output = check_output(f"yt-dlp --cookies ../youtube_cookie.txt -f 140 {video_url}", shell=True).decode("utf-8")
    for line in output.split("\n"):
        m = file_matcher.match(line)
        if m:
            output_file = m.group(1).strip()
            break

        m = file_matcher2.match(line)
        if m:
            output_file = m.group(1).strip()
            break

        m = file_matcher3.match(line)
        if m:
            output_file = m.group(1).strip()

    return output_file

ffmpeg_matcher = re.compile(r"\[segment @ [^\]]+\] Opening '(.*?)' for writing")

def transcribe_file(audio_file, segment_time=None):
    if segment_time is not None:
        # Segment it
        output = check_output(f"ffmpeg -i {audio_file} -map 0 -f segment -segment_time {segment_time} -reset_timestamps 1 -c copy \"tmp%03d.m4a\"", shell=True)
        # 
        
    with open(audio_file, "rb") as file:   
        response = openai.Audio.transcribe('whisper-1', file)
    return response['text']

if __name__ == "__main__":
    # Use argparse to get the video url
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_url", type=str, default=None)
    parser.add_argument("--audio_files", type=str, default=None, help="comma-separated audio files")
    parser.add_argument("--output", type=str, default="transcribed.txt")
    args = parser.parse_args()

    # save the transcribed text to a file
    with open(args.output, "w") as file:
        file.write(transcribe_youtube_video(args.video_url))
