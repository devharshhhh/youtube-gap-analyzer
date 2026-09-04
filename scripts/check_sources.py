from agents.youtube_agent import get_top_videos

videos = get_top_videos("scorpio vs scorpio n which one i should buy", max_results=8)

for v in videos:
    print(f"{v['title'][:60]} -> {v['content_source']} ({len(v['transcript'])} chars)")
