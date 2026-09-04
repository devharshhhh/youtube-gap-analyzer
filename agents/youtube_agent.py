import os
import tempfile
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()
youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

_whisper_model = None  # lazy-loaded, only created if actually needed


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


from concurrent.futures import ThreadPoolExecutor, as_completed

def get_top_videos(topic: str, max_results: int = 6) -> list:
    search_response = youtube.search().list(
        q=topic,
        part="snippet",
        type="video",
        order="relevance",
        maxResults=max_results,
    ).execute()

    items = search_response["items"]

    def process_video(item):
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        description = item["snippet"]["description"]
        channel = item["snippet"]["channelTitle"]

        content = get_video_content(video_id, title, description)

        return {
            "video_id": video_id,
            "title": title,
            "channel": channel,
            "transcript": content["text"],
            "content_source": content["source"],
        }

    videos = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_video, item) for item in items]
        for future in as_completed(futures):
            videos.append(future.result())

    return videos


def get_transcript(video_id: str) -> str:
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        try:
            transcript = transcript_list.find_transcript(['en'])
        except Exception:
            transcript = next(iter(transcript_list))
            if transcript.is_translatable:
                transcript = transcript.translate('en')

        fetched = transcript.fetch()
        return " ".join(snippet.text for snippet in fetched)

    except Exception:
        return ""


def transcribe_with_whisper(video_id: str) -> str:
    """
    Downloads audio only and transcribes it locally with Whisper.
    Works regardless of language or whether YouTube captions exist.
    """
    import yt_dlp

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path.replace(".mp3", ""),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }],
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception as e:
            print(f"yt-dlp download failed for {video_id}: {e}")
            return ""

        if not os.path.exists(audio_path):
            return ""

        model = _get_whisper_model()
        segments, info = model.transcribe(audio_path, beam_size=5)
        transcript = " ".join(segment.text for segment in segments)
        return transcript.strip()


def get_video_content(video_id: str, title: str, description: str) -> dict:
    transcript = get_transcript(video_id)
    if transcript:
        return {"text": transcript, "source": "transcript"}

    print(f"No caption transcript for {video_id}, trying Whisper...")
    whisper_transcript = transcribe_with_whisper(video_id)
    if whisper_transcript:
        return {"text": whisper_transcript, "source": "whisper_transcription"}

    return {"text": f"{title}. {description}", "source": "title_description"}
