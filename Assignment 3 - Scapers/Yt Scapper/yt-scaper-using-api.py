import googleapiclient.discovery

# Set up YouTube API
API_KEY = "YOUR_API_KEY"  
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def get_video_details(video_id):
    """Fetch video details like title, views, likes, and comments count."""
    youtube = googleapiclient.discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

    request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    response = request.execute()

    if "items" not in response or not response["items"]:
        print("Invalid Video ID or Video not found.")
        return None

    video = response["items"][0]
    details = {
        "Title": video["snippet"]["title"],
        "Channel": video["snippet"]["channelTitle"],
        "Views": video["statistics"].get("viewCount", "N/A"),
        "Likes": video["statistics"].get("likeCount", "N/A"),
        "Comments": video["statistics"].get("commentCount", "N/A")
    }

    return details

def get_video_comments(video_id, max_results=10):
    """Fetch top comments from a YouTube video."""
    youtube = googleapiclient.discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        order="relevance"
    )
    response = request.execute()

    comments = []
    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]
        comments.append({
            "Author": comment["authorDisplayName"],
            "Comment": comment["textDisplay"],
            "Likes": comment["likeCount"]
        })

    return comments

if __name__ == "__main__":
    video_url = input("Enter YouTube Video URL: ")
    video_id = video_url.split("v=")[-1].split("&")[0]  # Extract video ID from URL

    details = get_video_details(video_id)
    if details:
        print("\nVideo Details:")
        for key, value in details.items():
            print(f"{key}: {value}")

        print("\nTop Comments:")
        comments = get_video_comments(video_id)
        for i, comment in enumerate(comments, start=1):
            print(f"\nComment {i}:")
            print(f"Author: {comment['Author']}")
            print(f"Likes: {comment['Likes']}")
            print(f"Text: {comment['Comment']}")
