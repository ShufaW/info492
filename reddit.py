import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import re
from nltk.corpus import stopwords
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# Load the data
df_original = pd.read_csv("/Users/curttanaka/Downloads/archive/legacy/pse_isr_reddit_comments.csv")
df = df_original.copy()

# Filter for IsraelPalestine subreddit
israel_palestine_df = df[df['subreddit'] == 'IsraelPalestine'].copy()

class RedditSentimentAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.process_data()
    
    def process_data(self):
        # Convert timestamps
        self.df['created_time'] = pd.to_datetime(self.df['created_time'])
        
        # Clean text and add sentiment using self_text column
        self.df['clean_text'] = self.df['self_text'].fillna('').apply(self.clean_text)
        self.df['sentiment'] = self.df['clean_text'].apply(self.get_sentiment)
        self.df['sentiment_category'] = self.df['sentiment'].apply(self.categorize_sentiment)
    
    @staticmethod
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join(text.split())
    
    @staticmethod
    def get_sentiment(text):
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0
    
    @staticmethod
    def categorize_sentiment(score):
        if score > 0:
            return 'Positive'
        elif score < 0:
            return 'Negative'
        return 'Neutral'
    
    def generate_wordcloud(self):
        # Add custom stopwords
        custom_stopwords = set(['israel', 'palestine', 'israeli', 'palestinian'] + list(STOPWORDS))
        
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            stopwords=custom_stopwords,
            min_font_size=10
        ).generate(' '.join(self.df['clean_text']))
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.title('Most Common Words in r/IsraelPalestine Posts')
        plt.show()
    
    def plot_sentiment_distribution(self):
        # Create pie chart
        sentiment_counts = self.df['sentiment_category'].value_counts()
        
        plt.figure(figsize=(10, 8))
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        plt.pie(sentiment_counts,
                labels=sentiment_counts.index,
                colors=colors,
                autopct='%1.1f%%',
                explode=[0.03, 0.03, 0.08])
        plt.title('Sentiment Distribution in r/IsraelPalestine Posts', pad=20)
        plt.show()
        
        # Create funnel chart
        fig = go.Figure(go.Funnelarea(
            text=sentiment_counts.index,
            values=sentiment_counts.values,
            title={"position": "top center"}
        ))
        fig.update_layout(
            title="Sentiment Distribution Funnel Chart",
            title_x=0.5,
            width=500,
            height=400
        )
        fig.show()
    
    def plot_sentiment_over_time(self):
        # Resample by day and calculate mean sentiment
        daily_sentiment = self.df.set_index('created_time')['sentiment'].resample('D').mean()
        
        plt.figure(figsize=(15, 6))
        plt.plot(daily_sentiment.index, daily_sentiment.values, color='blue', alpha=0.6)
        plt.title('Average Sentiment Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def print_summary_stats(self):
        """Print summary statistics about the data"""
        print("\nSummary Statistics:")
        print("-" * 50)
        print(f"Total number of posts: {len(self.df)}")
        print(f"Date range: {self.df['created_time'].min()} to {self.df['created_time'].max()}")
        print("\nSentiment Distribution:")
        print(self.df['sentiment_category'].value_counts(normalize=True).round(3) * 100)
        print("\nAverage sentiment score:", round(self.df['sentiment'].mean(), 3))
        
    def get_top_words(self, n=10):
        """Get the most common words in the text"""
        from collections import Counter
        words = ' '.join(self.df['clean_text']).split()
        return Counter(words).most_common(n)

# Create the analyzer instance
analyzer = RedditSentimentAnalyzer(israel_palestine_df)

# Print summary statistics
analyzer.print_summary_stats()

print("\nMost common words:")
print("-" * 50)
for word, count in analyzer.get_top_words():
    print(f"{word}: {count}")

# Generate visualizations
analyzer.generate_wordcloud()
analyzer.plot_sentiment_distribution()
analyzer.plot_sentiment_over_time()