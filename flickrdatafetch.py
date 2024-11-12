import json
import requests
import time
import csv
from requests_oauthlib import OAuth1
from urllib.parse import parse_qs
from flickrapi import FlickrAPI
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

API_KEY = '6121804e178a34ebe49444e858987ee5'
API_SECRET = '0995d081c0eccf00'
API_END = 'https://api.flickr.com/services/rest/'
flickr = FlickrAPI(API_KEY, API_SECRET, format='parsed-json')

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def search_for_photos(keywords, num_images=10):
    photos = flickr.photos.search(text=keywords, per_page=num_images, sort='relevance')
    return [photo['id'] for photo in photos['photos']['photo']]

def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment, scores

# def fetch_and_display_comments(image_ids):
#     print("\n{:<15} {:<15} {:<20} {:<}".format(
#         "Photo ID", "Author", "Date", "Comment Text"))
#     print("-" * 80)

#     for photo_id in image_ids:
#         params = {
#             'method': 'flickr.photos.comments.getList',
#             'photo_id': photo_id,
#             'api_key': API_KEY,
#             'format': 'json',
#             'nojsoncallback': 1
#         }
#         response = requests.get(API_END, params=params)
#         if response.status_code == 200:
#             data = json.loads(response.text)
#             comments = data.get('comments', {}).get('comment', [])

#             if comments:  # Only print if there are comments
#                 for comment in comments:
#                     print("{:<15} {:<15} {:<20} {:<}".format(
#                         photo_id,
#                         comment.get('authorname', '')[:14],
#                         comment.get('datecreate', '')[:19],
#                         comment.get('_content', '')[:50] + ('...' if len(comment.get('_content', '')) > 50 else '')
#                     ))

def fetch_and_analyze_comments(image_ids):
    print("\n{:<15} {:<15} {:<20} {:<50} {:<10}".format(
        "Photo ID", "Author", "Date", "Comment Text", "Sentiment"))
    print("-" * 100)

    for photo_id in image_ids:
        params = {
            'method': 'flickr.photos.comments.getList',
            'photo_id': photo_id,
            'api_key': API_KEY,
            'format': 'json',
            'nojsoncallback': 1
        }
        response = requests.get(API_END, params=params)
        if response.status_code == 200:
            data = json.loads(response.text)
            comments = data.get('comments', {}).get('comment', [])

            if comments:  # Only print if there are comments
                for comment in comments:
                    comment_text = comment.get('_content', '')
                    sentiment, scores = analyze_sentiment(comment_text)
                    print("{:<15} {:<15} {:<20} {:<50} {:<10}".format(
                        photo_id,
                        comment.get('authorname', '')[:14],
                        comment.get('datecreate', '')[:19],
                        comment_text[:50] + ('...' if len(comment_text) > 50 else ''),
                        sentiment
                    ))

def main():
    search_word = 'palestine'
    image_ids = search_for_photos(search_word)
    # fetch_and_display_comments(image_ids)
    fetch_and_analyze_comments(image_ids)

if __name__ == "__main__":
    main()

# def fetch_and_save_comment(image_ids, query, output_file='output.csv'):
#     with open(output_file, mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Photo ID', 'Comment ID', 'Author', 'Date', 'Comment Text'])
#         for photo_id in image_ids:
#             params = {
#                 'method': 'flickr.photos.comments.getList',
#                 'photo_id': photo_id,
#                 'api_key': API_KEY,
#                 'format': 'json',
#                 'nojsoncallback': 1
#             }
#             response = requests.get(API_END, params=params)
#             if response.status_code == 200:
#                 data = json.loads(response.text)
#                 comments = data.get('comments', {}).get('comment', [])

#                 for comment in comments:
#                     comment_text = comment.get('_content', '')
#                     if query.lower() in comment_text.lower():
#                         writer.writerow([
#                             comment.get('id'),
#                             comment.get('authorname'),
#                             comment.get('datecreate'),
#                             comment_text
#                         ])
#             else:
#                 print("Can't find data for this image")




# photos = flickr.photos.search(text='landscape', per_page=5)
# for photo in photos.find('photos').findall('photo'):
#     print(photo.get('id'))

# photo_id = '54105892831'

# response_text = response.text
# print(response_text)
# data = json.loads(response.text)
# print(data)

# comments = data['comments']
# for comment in comments:
#     print(comment[1])


# API_KEY = '6121804e178a34ebe49444858987ee5'
# API_END = 'https://api.flickr.com/services/rest/'

# def fetch_flickr_comments(method, params):
#     params['method'] = method
#     params['api_key'] = API_KEY
#     params['format'] = json
#     params['nojsoncallback'] = 1

#     response = requests.get(API_END, params=params)
#     data = json.loads(response.text)
#     return data

# photo_id = '10289'
# params = {
#     'photo_id': photo_id
# }
# data = fetch_flickr_comments('flickr.activity.userComments', params)
# print(data)

# comments = data['photo']['comments']['_content']
# for comment in comments:
#     print(comment['author'], comment['_content'])

# # 1 - get request token
# request_token_url = "https://www.flickr.com/services/oauth/request_token"
# oauth = OAuth1(request_token_url)

# from requests_oauthlib import OAuth1
# from urllib.parse import parse_qs