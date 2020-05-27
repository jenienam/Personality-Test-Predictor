
import requests
import json
import GetOldTweets3 as got

tweeter = input()
count = 50

tweetCriteria = got.manager.TweetCriteria().setUsername(tweeter)                                        .setMaxTweets(count)
tweets = got.manager.TweetManager.getTweets(tweetCriteria)
users_tweets = []

for tweet in tweets:
    users_tweets.append(tweet.text)

obj = {}
obj['author'] = tweeter
obj['comments'] = users_tweets

with open("tweeter_comments.json", "w") as jsonFile:
    json.dump(obj, jsonFile)


