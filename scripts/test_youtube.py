import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
load_dotenv()
youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
response = youtube.search().list(
 q="python tutorial",
 part="snippet",
 type="video",
 maxResults=3
).execute()
for item in response["items"]:
 print(item["snippet"]["title"])
