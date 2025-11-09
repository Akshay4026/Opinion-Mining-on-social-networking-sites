import os
import googleapiclient.discovery
import pandas as pd
import streamlit as st

def get_youtube_comments(video_id, max_results):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyD1x6zgv9mkD68QGPIub2mJEvD_MY4dTHk"  # Replace with your own API key

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    comments = []

    next_page_token = None
    total_comments_retrieved = 0

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results - total_comments_retrieved, 100),
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['updatedAt'],
                comment['likeCount'],
                comment['textDisplay']
            ])
            total_comments_retrieved += 1

        if 'nextPageToken' in response and total_comments_retrieved < max_results:
            next_page_token = response['nextPageToken']
        else:
            break

    return comments

def main():
    st.title("YouTube Comments Downloader")

    video_id = st.text_input("Enter YouTube Video ID:")
    max_results = st.slider("Maximum Number of Comments to Retrieve:", min_value=1, max_value=1000, value=100)

    if st.button("Download Comments"):
        comments = get_youtube_comments(video_id, max_results)
        df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])

        # Specify the desired folder path
        folder_path = "D:/Major_final/data"  # Update the folder path as needed

        # Specify the file name
        file_name = "youtube_comments.csv"

        # Combine the folder path and file name
        file_path = os.path.join(folder_path, file_name)

        # Save the DataFrame to the CSV file
        df.to_csv(file_path, index=False)

        st.success(f"CSV file saved to: {file_path}")

if __name__ == "__main__":
    main()
