
import requests
import json

def getPushshiftData(author):
    url = f"https://api.pushshift.io/reddit/search/comment/?author={author}&sort=desc&size=50"
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']

input_info = input()

reddit_comments = []
author = input_info

data = getPushshiftData(author)

for comment in data:
    reddit_comments.append(comment['body'])
    data = getPushshiftData(author)

obj = {}
obj['author'] = author
obj['comments'] = reddit_comments

with open("reddit_comments.json", "w") as jsonFile:
    json.dump(obj, jsonFile)



