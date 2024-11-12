from pynytimes import NYTAPI
import os
import datetime
import json
import requests
import pandas as pd


nyt = NYTAPI("6PHpPgcrP9AlMU82J12ty8e6QaKfm8PU", parse_dates=True)
url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json?q=israel&api-key=6PHpPgcrP9AlMU82J12ty8e6QaKfm8PU'
r = requests.get(url)
json_data = r.json()
json_string = json.dumps(json_data)

# results = json_data.get("results")
# results_string = json.dumps(results)
# df = pd.read_json(results_string)
print(json_string)
# articles = nyt.article_search(
#     query = "Israeli–Palestinian Conflict",
#     results = 5,
#     # Search for articles in January and February 2019
#     dates = {
#         "begin": datetime.datetime(2018, 1, 30),
#         "end": datetime.datetime(2024, 8, 28)
#     },
#     options = {
#         "sort": "oldest", # Sort by oldest options
#         # Return articles from the following four sources
#         "sources": [
#             "New York Times",
#             "AP",
#             "Reuters",
#             "International Herald Tribune"
#         ],
#         # Only get information from the Politics desk
#         "news_desk": [
#             "Politics"
#         ],
#         # Only return ....
#         "type_of_material": [
#             "News Analysis"
#         ],
#         # The article text should contain the word..
#         "body": [
#             "death"
#         ],
#         # Headline should contain...
#         "headline": [
#             "conflict",
#             "war",
#             "toll"
#         ]
#     }
# )

# articles = nyt.article_search(query="Israeli–Palestinian Conflict", results=1)
# print(articles[1])
