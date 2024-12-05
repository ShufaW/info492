import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from flickrapi import FlickrAPI
import html
import re
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


nltk.download("vader_lexicon")
sid = SentimentIntensityAnalyzer()


API_KEY = "6121804e178a34ebe49444e858987ee5"
API_SECRET = "0995d081c0eccf00"
flickr = FlickrAPI(API_KEY, API_SECRET, format="parsed-json")


def clean_comment_text(text):
    text = html.unescape(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = " ".join(text.split()).strip()
    return text if len(text) > 3 else None


def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores["compound"] >= 0.05:
        sentiment = "Positive"
    elif scores["compound"] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, scores["compound"]


def fetch_comments(photo_ids):
    comment_data = []
    photo_comment_counts = {}
    for photo_id in photo_ids:
        try:
            response = flickr.photos.comments.getList(photo_id=photo_id)
            comments = response.get("comments", {}).get("comment", [])
            photo_comment_counts[photo_id] = len(comments)
            for comment in comments:
                comment_text = clean_comment_text(comment.get("_content", ""))
                if comment_text:
                    sentiment, score = analyze_sentiment(comment_text)
                    comment_data.append(
                        {
                            "photo_id": photo_id,
                            "author": comment.get("authorname", ""),
                            "date": datetime.fromtimestamp(
                                int(comment.get("datecreate", 0))
                            ).strftime("%Y-%m-%d"),
                            "comment_text": comment_text,
                            "sentiment": sentiment,
                            "sentiment_score": score,
                        }
                    )
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching comments for photo ID {photo_id}: {e}")
    return pd.DataFrame(comment_data), photo_comment_counts


def search_for_photos(keyword, start_date, end_date, num_images=100):
    start = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    photos = flickr.photos.search(
        tags=keyword,
        tag_mode="all",
        min_upload_date=start,
        max_upload_date=end,
        per_page=num_images,
        sort="date-posted-desc",
        extras="date_upload",
    )
    photo_data = []
    for photo in photos["photos"]["photo"]:
        photo_url = f"https://farm{photo['farm']}.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}.jpg"
        photo_data.append(
            {
                "id": photo["id"],
                "title": photo["title"],
                "url": photo_url,
            }
        )
    return photo_data


def plot_sentiment_trends(comments_df):
    comments_df["date"] = pd.to_datetime(comments_df["date"])

    sentiment_counts = (
        comments_df.groupby(["date", "sentiment"]).size().reset_index(name="count")
    )

    sentiment_pivot = sentiment_counts.pivot(
        index="date", columns="sentiment", values="count"
    ).fillna(0)

    plt.figure(figsize=(10, 6))
    plt.plot(
        sentiment_pivot.index,
        sentiment_pivot.get("Positive", 0),
        label="Positive",
        marker="o",
    )
    plt.plot(
        sentiment_pivot.index,
        sentiment_pivot.get("Neutral", 0),
        label="Neutral",
        marker="o",
    )
    plt.plot(
        sentiment_pivot.index,
        sentiment_pivot.get("Negative", 0),
        label="Negative",
        marker="o",
    )

    plt.title("Sentiment Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend(title="Sentiment")
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)


def highlight_emotion_keywords(text, emotion_words):
    """
    Highlight Sentiment Keywords
    :param text: comment text
    :param emotion_words: list of emotion keywords
    :return: HTML text with highlighting effect
    """
    for word in emotion_words:
        text = re.sub(
            f"\\b({re.escape(word)})\\b", r"<mark>\1</mark>", text, flags=re.IGNORECASE
        )
    return text


def analyze_sentiment_with_keywords(text):
    """
    Analyze the text sentiment and extract the keywords that contribute most to the sentiment score
    :param text: text of the input comments
    :return: sentiment type, sentiment score, sentiment keywords
    """

    scores = sid.polarity_scores(text)

    sentiment = "Neutral"
    if scores["compound"] >= 0.05:
        sentiment = "Positive"
    elif scores["compound"] <= -0.05:
        sentiment = "Negative"

    words = text.split()
    emotion_words = []
    for word in words:
        if word.lower() in sid.lexicon:
            emotion_words.append(word)

    return sentiment, scores["compound"], emotion_words


def main():
    st.title("Flickr Sentiment Analysis with Emotion Keywords")
    st.write(
        "Analyze public sentiment on Flickr comments over time based on specific keywords."
    )

    keyword = st.text_input("Enter keyword (e.g., 'Gaza', 'Palestine'):", "Gaza")
    start_date = st.date_input("Start Date:", datetime(2023, 1, 1))
    end_date = st.date_input("End Date:", datetime(2024, 11, 1))
    num_images = st.slider("Number of photos to analyze per keyword:", 10, 500, 100)

    if st.button("Run Analysis"):
        st.write(f"Searching photos for keyword: {keyword}")
        photos = search_for_photos(
            keyword,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            num_images,
        )

        if photos:
            st.write(f"Found {len(photos)} photos. Fetching comments for analysis...")
            photo_ids = [photo["id"] for photo in photos]
            comments_df, photo_comment_counts = fetch_comments(photo_ids)

            # Sort by the number of comments and keep the first four photos
            sorted_photos = sorted(
                photos, key=lambda x: photo_comment_counts.get(x["id"], 0), reverse=True
            )[:4]

            st.write("Displaying top 4 photos with most comments...")

            for i in range(0, len(sorted_photos), 2):
                cols = st.columns(2)
                for j, photo in enumerate(sorted_photos[i : i + 2]):
                    with cols[j]:
                        st.image(
                            photo["url"],
                            caption=f"{photo['title']} ({photo_comment_counts.get(photo['id'], 0)} comments)",
                            use_column_width=True,
                        )

                        with st.expander(
                            f"View Comments for '{photo['title']}'", expanded=False
                        ):
                            photo_comments = comments_df[
                                comments_df["photo_id"] == photo["id"]
                            ]
                            st.write("### Comments:")
                            for _, comment in photo_comments.iterrows():
                                sentiment, score, emotion_words = (
                                    analyze_sentiment_with_keywords(
                                        comment["comment_text"]
                                    )
                                )
                                highlighted_comment = highlight_emotion_keywords(
                                    comment["comment_text"], emotion_words
                                )
                                st.markdown(
                                    f"- **{comment['author']}**: {highlighted_comment} "
                                    f"(Sentiment: {sentiment}, Score: {score:.2f})",
                                    unsafe_allow_html=True,
                                )

            if not comments_df.empty:
                st.success("Analysis complete!")
                st.write("### Sentiment Analysis Results")
                st.dataframe(comments_df)

                st.write("### Sentiment Trends Over Time")
                plot_sentiment_trends(comments_df)
            else:
                st.warning("No comments found for the selected photos.")
        else:
            st.error("No photos found for the given keyword and date range.")


if __name__ == "__main__":
    main()
