#lets build a twitter scapper
# there is issues with scapting these sites which reqired login
# twtitter is not open sources so it will require to login to get access to more data witch makes the data scaping hard

print('Twitter Scapper')
#for it there are many library 
import tweepy
import pandas as pd

# Twitter API credentials
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
ACCESS_TOKEN = "your_access_token"
ACCESS_SECRET = "your_access_secret"

# Authenticate to Twitter
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Function to scrape tweets
def scrape_tweets(query, count=100):
    tweets_data = []
    
    for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode="extended").items(count):
        tweets_data.append({
            "Tweet ID": tweet.id,
            "Username": tweet.user.screen_name,
            "Followers": tweet.user.followers_count,
            "Tweet": tweet.full_text,
            "Date": tweet.created_at,
            "Retweets": tweet.retweet_count,
            "Likes": tweet.favorite_count,
            "User Location": tweet.user.location
        })

    # Save to CSV
    df = pd.DataFrame(tweets_data)
    df.to_csv("tweets_data.csv", index=False)
    print("Tweets saved to 'tweets_data.csv'")

# Example: Scrape tweets containing "AI" (change the keyword as needed)
scrape_tweets("AI", count=50)
