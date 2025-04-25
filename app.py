# app.py - Main Streamlit application with Twitter API integration

import streamlit as st
import pandas as pd
import plotly.express as px
import time
import requests
from datetime import datetime
import threading
from collections import deque
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Import Twitter integration if available
try:
    from twitter_stream import TwitterStream
    TWITTER_AVAILABLE = True
except Exception as e:
    TWITTER_AVAILABLE = False
    st.warning(f"Twitter integration not available: {str(e)}. Using simulation mode.")

# Mock Groq API for sentiment analysis (replace with actual Groq API in production)
class GroqSentimentAnalyzer:
    def __init__(self):
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.api_key = os.environ.get("GROQ_API_KEY")
        
        if self.api_key:
            st.success("Groq API key detected. Using real Groq inference.")
        else:
            st.info("No Groq API key found. Using simulated sentiment analysis.")
        
        print(f"Initializing sentiment analysis with {self.model_name} on Groq...")
        
    def analyze(self, text):
        """Analyze sentiment using Groq API or simulation"""
        if self.api_key:
            return self._analyze_with_groq(text)
        else:
            return self._simulate_sentiment(text)
    
    def _analyze_with_groq(self, text):
        """Real sentiment analysis using Groq API"""
        # Note: Implementation depends on Groq's specific API
        # This is a placeholder - replace with actual Groq implementation
        try:
            # Simulate API call for now
            time.sleep(0.1)
            
            # TODO: Replace with actual Groq API call
            # Example:
            # response = requests.post(
            #     "https://api.groq.com/sentiment",
            #     headers={"Authorization": f"Bearer {self.api_key}"},
            #     json={"text": text}
            # )
            # result = response.json()
            # return {
            #     "sentiment": result["label"],
            #     "confidence": result["score"]
            # }
            
            # Until the above is implemented, fall back to simulation
            return self._simulate_sentiment(text)
            
        except Exception as e:
            st.error(f"Error with Groq API: {str(e)}")
            return {"sentiment": "neutral", "confidence": 0.5}
    
    def _simulate_sentiment(self, text):
        """Simulate sentiment analysis for demo purposes"""
        # Simple rule-based sentiment simulation
        positive_words = ['great', 'awesome', 'excellent', 'good', 'love', 'amazing', 'faster', 'incredible']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'disappointed', 'issues', 'error', 'frustrated']
        
        text_lower = text.lower()
        
        # Count sentiment words
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Determine base sentiment
        if pos_count > neg_count:
            sentiment = "positive"
            base_confidence = 0.7 + (pos_count - neg_count) * 0.05
        elif neg_count > pos_count:
            sentiment = "negative"
            base_confidence = 0.7 + (neg_count - pos_count) * 0.05
        else:
            sentiment = "neutral"
            base_confidence = 0.7
        
        # Normalize confidence to 0.7-0.98 range
        confidence = min(0.98, max(0.7, base_confidence))
        
        return {
            "sentiment": sentiment,
            "confidence": confidence
        }

# Tweet simulator for the demo (used when Twitter API is not available)
class TweetSimulator:
    def __init__(self):
        self.hashtag = "#GrowWithGroqHack"
        self.example_tweets = [
            f"Just deployed my first ML model using Groq! So much faster than anything I've used before. {self.hashtag}",
            f"The developer experience with Groq is incredible. Game changer for my projects! {self.hashtag}",
            f"Having some issues with the Groq API documentation. Anyone else facing this? {self.hashtag}",
            f"Wow! My inference speed went from 2s to 0.1s after switching to Groq. {self.hashtag} #ML",
            f"Not sure if Groq is worth the hype yet. Still evaluating... {self.hashtag}",
            f"Absolutely blown away by Groq's performance on transformer models! {self.hashtag}",
            f"{self.hashtag} is trending and for good reason. The speed is unmatched!",
            f"Frustrated with the current limitations of Groq's model selection. {self.hashtag}",
            f"Can't get my code to work with Groq. Any help would be appreciated. {self.hashtag} #coding",
            f"Just compared benchmark results - Groq is 20x faster than what I was using before! {self.hashtag}",
            f"Attending the {self.hashtag} workshop today. Looking forward to learning more!",
            f"The pricing for Groq seems reasonable given the performance gains. {self.hashtag}",
            f"Not happy with customer support response times from Groq team. {self.hashtag}",
            f"Just hit a roadblock integrating Groq with my existing infrastructure. Any tips? {self.hashtag}",
            f"This is revolutionary technology! Can't believe how snappy everything feels with Groq. {self.hashtag}",
            f"Meh, I expected more from Groq based on the hype. {self.hashtag}",
            f"Our team has fully migrated to Groq and we're seeing massive throughput improvements! {self.hashtag}",
            f"Error after error trying to use Groq's API. Really disappointed. {self.hashtag}",
            f"The future of AI inference is here and it's spelled G-R-O-Q! {self.hashtag}",
            f"Halfway through the hackathon and Groq is making everything so much easier! {self.hashtag}"
        ]
    
    def get_random_tweet(self):
        """Generate a random tweet with timestamp and user"""
        import random
        
        tweet = random.choice(self.example_tweets)
        username = f"@user_{random.randint(1000, 9999)}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "username": username,
            "text": tweet,
            "timestamp": timestamp
        }

# Slack integration for alerts
class SlackAlerter:
    def __init__(self, webhook_url=None):
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL") or "https://hooks.slack.com/services/REPLACE/WITH/YOUR_WEBHOOK_URL"
        
    def send_alert(self, message):
        """Send alert to Slack channel"""
        if self.webhook_url.startswith("https://hooks.slack.com/services/REPLACE"):
            # Demo mode - just print the alert
            st.warning(f"SLACK ALERT (DEMO MODE): {message}")
            return {"status": "demo_mode", "message": "Alert displayed in demo mode"}
        
        # In production, this would send an actual request to Slack
        payload = {
            "text": message
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload)
            if response.status_code == 200:
                st.success("Alert sent to Slack successfully")
            else:
                st.error(f"Failed to send alert to Slack: {response.status_code}")
            return {"status": "sent", "response": response.text}
        except Exception as e:
            st.error(f"Error sending alert to Slack: {str(e)}")
            return {"status": "error", "message": str(e)}

# Initialize session state variables if they don't exist
def initialize_session_state():
    if 'tweets' not in st.session_state:
        st.session_state.tweets = []
    
    if 'sentiment_counts' not in st.session_state:
        st.session_state.sentiment_counts = {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }
    
    if 'recent_negative' not in st.session_state:
        st.session_state.recent_negative = deque(maxlen=20)
    
    if 'word_counts' not in st.session_state:
        st.session_state.word_counts = {}
    
    if 'stream_active' not in st.session_state:
        st.session_state.stream_active = False
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = GroqSentimentAnalyzer()
    
    if 'tweet_source' not in st.session_state:
        if TWITTER_AVAILABLE:
            try:
                st.session_state.tweet_source = TwitterStream()
                st.session_state.using_real_twitter = True
            except Exception as e:
                st.error(f"Error initializing Twitter API: {str(e)}. Falling back to simulation.")
                st.session_state.tweet_source = TweetSimulator()
                st.session_state.using_real_twitter = False
        else:
            st.session_state.tweet_source = TweetSimulator()
            st.session_state.using_real_twitter = False
    
    if 'slack_alerter' not in st.session_state:
        st.session_state.slack_alerter = SlackAlerter()
    
    if 'hashtag' not in st.session_state:
        st.session_state.hashtag = "#GrowWithGroqHack"

# Function to check if we need to send a slack alert
def check_for_alert_condition():
    # Get negative tweets from the last 10 seconds
    current_time = time.time()
    recent_negative_count = sum(1 for t in st.session_state.recent_negative 
                               if current_time - t <= 10)
    
    if recent_negative_count >= 3:
        message = f"⚠️ Negative sentiment spike detected! Check tweets for {st.session_state.hashtag}."
        st.session_state.slack_alerter.send_alert(message)
        # Clear the queue to avoid repeated alerts
        st.session_state.recent_negative.clear()

# Function to update word counts from tweet text
def update_word_counts(text):
    # Clean and tokenize the text
    text = re.sub(r'http\S+', '', text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    
    # Filter out common stop words (simplified version)
    stop_words = {'the', 'a', 'an', 'and', 'is', 'in', 'it', 'to', 'with', 'for', 'of', 'on', 'at', 'by', 'from', 'up', 'about', 'into', 'over', 'after'}
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Update word counts
    for word in words:
        if word in st.session_state.word_counts:
            st.session_state.word_counts[word] += 1
        else:
            st.session_state.word_counts[word] = 1

# Function to process a single tweet
def process_tweet():
    if not st.session_state.stream_active:
        return
    
    # Get a tweet - either from Twitter API or simulator
    if st.session_state.using_real_twitter:
        tweet = st.session_state.tweet_source.get_stream_sample()
        if not tweet:  # Fall back to simulation if no tweets are available
            tweet = TweetSimulator().get_random_tweet()
    else:
        tweet = st.session_state.tweet_source.get_random_tweet()
    
    # Analyze sentiment
    result = st.session_state.analyzer.analyze(tweet["text"])
    sentiment = result["sentiment"]
    confidence = result["confidence"]
    
    # Update sentiment counts
    st.session_state.sentiment_counts[sentiment] += 1
    
    # Track negative tweets for alert condition
    if sentiment == "negative":
        st.session_state.recent_negative.append(time.time())
    
    # Check if we need to send an alert
    check_for_alert_condition()
    
    # Update word counts for word cloud
    update_word_counts(tweet["text"])
    
    # Add processed tweet to the list
    tweet_with_sentiment = {
        **tweet,
        "sentiment": sentiment,
        "confidence": confidence
    }
    st.session_state.tweets.insert(0, tweet_with_sentiment)
    
    # Keep only the 100 most recent tweets
    if len(st.session_state.tweets) > 100:
        st.session_state.tweets = st.session_state.tweets[:100]

# Function to continuously process tweets
def tweet_stream():
    while st.session_state.stream_active:
        process_tweet()
        time.sleep(1.5)  # Process a new tweet every 1.5 seconds

# Main function to render the Streamlit app
def main():
    st.set_page_config(
        page_title="Live Tweet Sentiment Meter",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Title and description
    st.title("📊 Live Tweet Sentiment Meter")
    st.markdown(f"Real-time sentiment analysis of tweets with **{st.session_state.hashtag}** hashtag")
    
    # Data source indicator
    if st.session_state.using_real_twitter:
        st.success("✅ Using real Twitter API data")
    else:
        st.info("ℹ️ Using simulated tweet data (Twitter API not configured)")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Hashtag input
        new_hashtag = st.text_input(
            "Hashtag to track",
            value=st.session_state.hashtag
        )
        
        if new_hashtag != st.session_state.hashtag:
            st.session_state.hashtag = new_hashtag
            if st.session_state.using_real_twitter:
                try:
                    st.session_state.tweet_source = TwitterStream(hashtag=new_hashtag)
                except Exception as e:
                    st.error(f"Error updating Twitter stream: {str(e)}")
        
        # Webhook URL input
        webhook_url = st.text_input(
            "Slack Webhook URL",
            value=st.session_state.slack_alerter.webhook_url,
            type="password"
        )
        
        if webhook_url != st.session_state.slack_alerter.webhook_url:
            st.session_state.slack_alerter = SlackAlerter(webhook_url)
        
        st.header("Controls")
        
        # Start/Stop button
        if not st.session_state.stream_active:
            if st.button("▶️ Start Stream"):
                st.session_state.stream_active = True
                # Start the stream in a background thread
                thread = threading.Thread(target=tweet_stream)
                thread.daemon = True
                thread.start()
        else:
            if st.button("⏹️ Stop Stream"):
                st.session_state.stream_active = False
        
        # Reset button
        if st.button("🔄 Reset Data"):
            st.session_state.tweets = []
            st.session_state.sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            st.session_state.recent_negative.clear()
            st.session_state.word_counts = {}
        
        # Display alert rules
        st.subheader("Alert Rules")
        st.info(f"⚠️ Alert will be sent to Slack if 3 or more negative tweets are detected within 10 seconds")
        
        # Display credits
        st.markdown("---")
        st.markdown("### About")
        st.markdown("Live Tweet Sentiment Meter powered by:")
        st.markdown("- 🚀 Groq for fast inference")
        st.markdown("- 🐦 Twitter API for real-time data")
        st.markdown("- 📊 Streamlit for visualization")
        st.markdown("- 📱 Slack for alerts")

    # Main content area - split into columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Overall mood emoji
        total_tweets = sum(st.session_state.sentiment_counts.values())
        if total_tweets > 0:
            pos_percent = st.session_state.sentiment_counts["positive"] / total_tweets
            neg_percent = st.session_state.sentiment_counts["negative"] / total_tweets
            
            mood_emoji = "😐"  # Default neutral
            if pos_percent > 0.5:
                mood_emoji = "😊"
            elif neg_percent > 0.4:
                mood_emoji = "😠"
            
            overall_mood = f"Overall Mood: {mood_emoji}"
            st.subheader(overall_mood)
        else:
            st.subheader("Overall Mood: Waiting for data...")
        
        # Sentiment distribution chart
        st.subheader("Sentiment Distribution")
        sentiment_df = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [
                st.session_state.sentiment_counts["positive"],
                st.session_state.sentiment_counts["neutral"],
                st.session_state.sentiment_counts["negative"]
            ]
        })
        
        colors = {'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}
        
        # Create pie chart
        if total_tweets > 0:
            fig = px.pie(
                sentiment_df, 
                values='Count', 
                names='Sentiment',
                color='Sentiment',
                color_discrete_map=colors,
                hole=0.4
            )
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Start the stream to see sentiment distribution")
        
        # Word Cloud visualization
        st.subheader("Trending Words")
        if st.session_state.word_counts:
            # Generate word cloud
            wordcloud = WordCloud(
                width=600, 
                height=300, 
                background_color='white',
                colormap='viridis',
                max_words=50
            ).generate_from_frequencies(st.session_state.word_counts)
            
            # Display word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Start the stream to see trending words")
    
    with col2:
        # Latest tweets with sentiment labels
        st.subheader("Latest Tweets")
        
        if st.session_state.tweets:
            for tweet in st.session_state.tweets[:10]:  # Show only 10 most recent
                sentiment = tweet["sentiment"]
                
                # Set color based on sentiment
                if sentiment == "positive":
                    container_color = "#e6f4ea"  # Light green
                    border_color = "#4CAF50"  # Green
                elif sentiment == "neutral":
                    container_color = "#fff9e6"  # Light yellow
                    border_color = "#FFC107"  # Yellow
                else:  # negative
                    container_color = "#fdedeb"  # Light red
                    border_color = "#F44336"  # Red
                
                # Display tweet with styled container
                container = st.container(border=True)
                with container:
                    # Apply custom CSS for styled tweet container
                    custom_css = f"""
                    <style>
                    .tweet-container {{
                        background-color: {container_color};
                        border-left: 4px solid {border_color};
                        padding: 10px;
                        margin-bottom: 10px;
                        border-radius: 4px;
                    }}
                    .tweet-header {{
                        font-weight: bold;
                        margin-bottom: 5px;
                    }}
                    .tweet-timestamp {{
                        color: #657786;
                        font-size: 0.8em;
                    }}
                    .tweet-sentiment {{
                        font-weight: bold;
                        color: {border_color};
                        text-transform: uppercase;
                        font-size: 0.8em;
                    }}
                    </style>
                    """
                    
                    st.markdown(custom_css, unsafe_allow_html=True)
                    
                    # Display tweet content with styling
                    tweet_html = f"""
                    <div class="tweet-container">
                        <div class="tweet-header">{tweet["username"]}</div>
                        <div>{tweet["text"]}</div>
                        <div class="tweet-timestamp">{tweet["timestamp"]} · 
                        <span class="tweet-sentiment">{sentiment.upper()} ({tweet["confidence"]:.2f})</span></div>
                    </div>
                    """
                    
                    st.markdown(tweet_html, unsafe_allow_html=True)
        else:
            st.info("Start the stream to see tweets")

if __name__ == "__main__":
    main()