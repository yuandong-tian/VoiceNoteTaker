# Transcribe youtube video to text

import argparse
import os
import re
import openai
from subprocess import check_output

openai.api_key = os.environ.get("OPENAI_API_KEY")

file_matcher = re.compile(r"\[ffmpeg\] Correcting container in \"(.*?)\"") 

def transcribe_youtube_video(video_url: str) -> str:
    # First download the video using yt-dlp
    os.system(f"yt-dlp -o audio.mp3 {video_url}")

    output = check_output(f"python -m youtube_dl -f 140 \"{video_url}\"", shell=True).decode("utf-8")
    for line in output.split("\n"):
        m = file_matcher.match(line)
        if m:
            output_file = m.group(1).strip()

    with open(output_file, "rb") as file:   
        response = openai.Audio.transcribe('whisper-1', file)
    return response['text']

if __name__ == "__main__":
    # Use argparse to get the video url
    parser = argparse.ArgumentParser()
    parser.add_argument("video_url", type=str)
    parser.add_argument("--output", type=str, default="transcribed.txt")
    args = parser.parse_args()

    # save the transcribed text to a file
    with open(args.output, "w") as file:
        file.write(transcribe_youtube_video(args.video_url))
