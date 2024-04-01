"""
Downloads and transcribes the audio from YouTube's George Hotz Archive's tinygrad playlist (https://www.youtube.com/playlist?list=PLzFUMGbVxlQsh0fFZ2QKOBY25lz04A3hi).
Inspired by Karpathy's Lexicap (https://karpathy.ai/lexicap/).
"""

import json
import subprocess

from youtubesearchpython import Playlist
import whisper


def main():
    # We need a Playlist because it looks like Channel does not implement .getVideos()
    playlist_videos = Playlist.getVideos("https://www.youtube.com/playlist?list=PLzFUMGbVxlQsh0fFZ2QKOBY25lz04A3hi")
    last_video = playlist_videos.get("videos", [])[-1]

    print(last_video)
    with open(f"data/{last_video['id']}.json", "w") as f:
        f.write(json.dumps(last_video, sort_keys=True, indent=2))

    # Had to pip install yt-dlp: python3 -m pip install --no-deps -U yt-dlp
    # And then download ffmpeg for webm to mp3 transformation: https://github.com/yt-dlp/FFmpeg-Builds#ffmpeg-static-auto-builds 
    ytdlp_cmd = f"yt-dlp -x --audio-format mp3 --ffmpeg-location ./ffmpeg-master-latest-linux64-gpl/bin/ffmpeg -o ./data/{last_video['id']}.mp3 -- {last_video['id']}"
    print(f"running command {ytdlp_cmd}")
    subprocess.run(ytdlp_cmd.split(" "))

    print("Transcribing")
    # Had to install openai-whisper: pip install -U openai-whisper
    # Had to install ffmpeg: sudo apt update && sudo apt install ffmpeg
    model = whisper.load_model("small")
    result = model.transcribe(f"./data/{last_video['id']}.mp3")

    with open(f"data/{last_video['id']}.txt", "w") as f:
        f.write(result["text"])


if __name__ == "__main__":
    main()