# Enter app store id here: 
# https://itunes.apple.com/us/app/name/id{app_store_id}?mt=8
app_store_id = '334989259'
import pprint
import time
import typing
import csv
import requests
import sys
import numpy as np
import pandas as pd

def is_error_response(http_response, seconds_to_sleep: float = 1) -> bool:
    if http_response.status_code == 503:
        time.sleep(seconds_to_sleep)
        return False

    return http_response.status_code != 200


def get_json(url) -> typing.Union[dict, None]:
    response = requests.get(url)
    if is_error_response(response):
        return None
    json_response = response.json()
    return json_response

def write_to_csv(row_array, app_id):
    csv_name = 'data/raw/app_store_reviews_%s_%s.csv' % (app_id, time.time())
    print('Saved to: ' + csv_name)
    with open(csv_name, mode='w') as csv_file:
        fieldnames = ['review_id', 'title', 'author', 'author_url', 'version', 'rating', 'review', 'vote_count']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        for row in row_array:
            writer.writerow(row)


def get_reviews(app_id, page=1) -> typing.List[dict]:
    reviews: typing.List[dict] = []

    while True:
        url = ('https://itunes.apple.com/rss/customerreviews/id=%s/page=%s/sortby=mostrecent/json' % (app_id, page))
        json = get_json(url)

        if not json:
            return reviews

        data_feed = json.get('feed')

        if not data_feed.get('entry'):
            get_reviews(app_id, page + 1)

        reviews += [
            {
                'review_id': entry.get('id').get('label'),
                'title': entry.get('title').get('label'),
                'author': entry.get('author').get('name').get('label'),
                'author_url': entry.get('author').get('uri').get('label'),
                'version': entry.get('im:version').get('label'),
                'rating': entry.get('im:rating').get('label'),
                'review': entry.get('content').get('label'),
                'vote_count': entry.get('im:voteCount').get('label')
            }
            for entry in data_feed.get('entry')
            if not entry.get('im:name')
        ]

        page += 1


review_dict = get_reviews(app_store_id)
    
print('Found ' + str(len(review_dict)) + ' reviews')
# write_to_csv(reviews, app_store_id)
# df = pd.read_csv(reviews) # for use with csv
df = pd.DataFrame(review_dict)

titles = np.array(df['title'])
reviews = np.array(df['review'])
ratings = np.array(df['rating'])
versions = np.array(df['version'])
size = (len(df))
print('Review count: ' + str(size))
print('Versions count: ' + str(len(np.unique(versions, return_counts=False))))
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
pos_reviews = ''
neg_reviews = ''
concat_reviews = ''
score_arr = []
for i in range(size):
    sentence = reviews[i]
    concat_reviews += ' %s' % sentence
    vs = analyzer.polarity_scores(sentence)
    if vs.get('compound') >= 0:
        pos_reviews += ' %s' % sentence
    else:
        neg_reviews += ' %s' % sentence
    score_arr.append(vs.get('compound'))
    
scores = np.vstack([versions,score_arr])
print(str(scores[1][:5]) + ' ...')
import matplotlib.pyplot as plt

# Prepare plot
unique_versions = np.unique(scores[0], return_counts=True)[0] # Versions
unique_review_count = np.unique(scores[0], return_counts=True)[1]
sum_arr = []
sum_vers = []
for version in unique_versions:
    version_sum = 0
    count = 0
    for i in range(len(scores[1])):
        if version == scores[0][i]:
            version_sum += scores[1][i]
            count += 1;
            
    sum_arr.append(version_sum / count )#unique_review_count[np.where(unique_versions==version)[0][0]] also works
    sum_vers.append(version)
    
print(sum_vers)
print(sum_arr)
plt.figure(figsize=(20,8))
plt.bar(sum_vers, sum_arr, align='center', alpha=0.5)

plt.xlabel("Version",fontsize=16)
plt.ylabel("$Sentiment$",fontsize=16)
plt.title("Sentiment per Version - App Store Reviews")
plt.ylim(-1, 1)

plt.show()
plt.figure(figsize=(20,8))
plt.bar(sum_vers, unique_review_count, align='center', alpha=0.5)

plt.xlabel("Version",fontsize=16)
plt.ylabel("$Count$",fontsize=16)
plt.title("Count per Version - App Store Reviews")

plt.show()
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
stopwords.add('app')

def generate_wordcloud(text): # optionally add: stopwords=STOPWORDS and change the arg below
    wordcloud = WordCloud(relative_scaling = 1.0,
                          scale=3,
                          stopwords = stopwords
                          ).generate(text)
    plt.figure(figsize=(20,20))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

generate_wordcloud(neg_reviews)
generate_wordcloud(pos_reviews)