# twitter_stream.py

import tweepy
import os
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TwitterStream:
    def __init__(self, hashtag="#GrowWithGroqHack"):
        """Initialize the Twitter stream with API credentials and target hashtag"""
        
        # Get credentials from environment variables for security
        self.api_key = os.environ.get("TWITTER_API_KEY")
        self.api_secret = os.environ.get("TWITTER_API_SECRET")
        self.access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
        self.access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")
        
        # Validate credentials are available
        self._validate_credentials()
        
        # Initialize Twitter client
        self.client = self._initialize_client()
        
        # Search parameters
        self.hashtag = hashtag
        self.query = f"{hashtag} -is:retweet"  # Exclude retweets
        
        logging.info(f"TwitterStream initialized for hashtag: {hashtag}")
    
    def _validate_credentials(self):
        """Validate that all required credentials are available"""
        missing = []
        
        if not self.api_key:
            missing.append("TWITTER_API_KEY")
        if not self.api_secret:
            missing.append("TWITTER_API_SECRET")
        if not self.access_token:
            missing.append("TWITTER_ACCESS_TOKEN")
        if not self.access_token_secret:
            missing.append("TWITTER_ACCESS_TOKEN_SECRET")
        
        if missing:
            raise ValueError(f"Missing required Twitter API credentials: {', '.join(missing)}")
    
    def _initialize_client(self):
        """Initialize and return the Tweepy client"""
        try:
            # Create OAuth 1.0a authentication handler
            auth = tweepy.OAuth1UserHandler(
                self.api_key, 
                self.api_secret,
                self.access_token,
                self.access_token_secret
            )
            
            # Create API client
            client = tweepy.API(auth)
            
            # Test the credentials
            client.verify_credentials()
            logging.info("Twitter API credentials verified successfully")
            
            return client
        
        except Exception as e:
            logging.error(f"Error initializing Twitter client: {str(e)}")
            raise
    
    def get_recent_tweets(self, count=10):
        """Fetch recent tweets matching the hashtag"""
        try:
            # Search for tweets
            tweets = self.client.search_tweets(
                q=self.query,
                count=count,
                tweet_mode="extended",
                result_type="recent"
            )
            
            # Process tweets into standardized format
            processed_tweets = []
            
            for tweet in tweets:
                processed_tweet = {
                    "id": tweet.id,
                    "username": f"@{tweet.user.screen_name}",
                    "text": tweet.full_text,
                    "timestamp": tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "user_followers": tweet.user.followers_count,
                    "retweet_count": tweet.retweet_count,
                    "favorite_count": tweet.favorite_count
                }
                processed_tweets.append(processed_tweet)
            
            logging.info(f"Fetched {len(processed_tweets)} tweets with hashtag {self.hashtag}")
            return processed_tweets
            
        except Exception as e:
            logging.error(f"Error fetching tweets: {str(e)}")
            return []
    
    def get_stream_sample(self):
        """Get a single tweet from the stream - can be called repeatedly for streaming effect"""
        tweets = self.get_recent_tweets(count=5)
        
        if not tweets:
            return None
        
        # Return most recent tweet first
        return tweets[0] if tweets else None