import json
import requests
import time
import csv  # Import the csv module
from requests_oauthlib import OAuth1
from urllib.parse import parse_qs
from flickrapi import FlickrAPI

API_KEY = '6121804e178a34ebe49444e858987ee5'
API_SECRET = '0995d081c0eccf00'
API_END = 'https://api.flickr.com/services/rest/'

flickr = FlickrAPI(API_KEY, API_SECRET, format='parsed-json')

# Function to fetch photos and comments
def fetch_photos_and_comments(query):
    page = 1
    while True:
        photos = flickr.photos.search(text=query,
                                      per_page=100,  # Reasonable number to balance API calls and data volume
                                      extras='url_c, date_taken, owner_name, geo, tags',
                                      page=page)
        if 'photos' in photos and 'photo' in photos['photos']:
            if not photos['photos']['photo']:
                break  # No more photos available
            for photo in photos['photos']['photo']:
                photo_id = photo['id']
                # Fetch comments for each photo
                comments_response = flickr.photos.comments.getList(photo_id=photo_id)
                comments = []
                if 'comments' in comments_response and 'comment' in comments_response['comments']:
                    comments3 = [comment['_content'] for comment in comments_response['comments']['comment']]

                # Yield photo details including comments
                yield {
                    'Photo ID': photo_id,
                    'URL': photo.get('url_c'),
                    'Date Taken': photo['datetaken'],
                    'Owner Name': photo['ownername'],
                    'Geo': f"{photo.get('latitude', '')},{photo.get('longitude', '')}",
                    'Tags': photo['tags'],
                    'Comments': " | ".join(comments)  # Concatenate comments
                }
            page += 1
        else:
            break  # Stop if there's an error or no data
        time.sleep(1)  # Respect API rate limits

with open('flickr_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['Photo ID', 'URL', 'Date Taken', 'Owner Name', 'Geo', 'Tags', 'Comments'])
    writer.writeheader()
    for photo_info in fetch_photos_and_comments('Palestine'):
        writer.writerow(photo_info)