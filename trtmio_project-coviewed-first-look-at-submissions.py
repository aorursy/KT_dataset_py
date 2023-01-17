!pip install -U yarl
import os
import yarl
import numpy as np
import pandas as pd
import datetime as dt
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt
count_f = 0
frames = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in sorted(filenames):
        if filename.endswith('.tsv'):
            fname = os.path.join(dirname, filename)
            df = pd.read_csv(fname, sep='\t')
            frames.append(df)
            print("%4i"%count_f, "|", "%6i"%len(df), fname.split('/')[-1])
            count_f+=1
DATA = pd.concat(frames)
len(DATA)
DATA.sample(n=min(len(DATA),5))
exclude_urls = ['reddit.','redd.','youtube','youtu.','twitter','facebook','fb','google.','.mp3','.pdf','/r/','imgur.']
exclude_urls
submission_datetime = []
url_hosts = []
for created_utc, url in DATA[['created_utc','url']].values.tolist():
    datetime = dt.datetime.fromtimestamp(int(created_utc))
    submission_datetime.append(datetime)
    try:
        S = set([x in url for x in exclude_urls])
        if len(S)==1 and not True in S:
            url = yarl.URL(url)
            url_hosts.append(url.host)
    except:
        print(url)
print(len(submission_datetime), len(url_hosts), min(submission_datetime), max(submission_datetime), sep='\n')
TOP_N_URL_HOSTS = 25

C = Counter(url_hosts)
for count_u, (url_host, count_host) in enumerate(sorted(C.items(), key=itemgetter(1), reverse=True)):
    print("%6i"%count_host, url_host)
    if count_u>=TOP_N_URL_HOSTS:
        break
DATA['date'] = DATA.apply(lambda row: dt.date.fromtimestamp(int(row.created_utc)), axis=1)
D = DATA[['id','date']].groupby('date').count().sort_values('date', ascending=True)
X, X_label, Y = [], [], []
for count_d, (date, count_id) in enumerate(zip(D.index, D.id)):
    X.append(count_d)
    Y.append(count_id)
    X_label.append(date.strftime('%Y-%m-%d'))
len(X), len(Y), len(X_label)
[i for i in range(len(X_label))][::7]
plt.figure(figsize=(15,6))
plt.scatter(X, Y)
plt.plot(X, Y)
for w in [i for i in range(len(X_label))][::7]:
    plt.vlines(w, min(Y)-0.1*max(Y), max(Y)+0.1*max(Y), color='k', alpha=0.25)
plt.xticks([i for i in range(len(X_label))][::7], X_label[::7], rotation=60)
plt.xlim(-1,len(X)+1)
plt.ylim(0, max(Y)+0.1*max(Y))
plt.title("News Articles submitted to /r/Coronavirus per Day")
plt.xlabel('date')
plt.ylabel('submission count')
plt.show()
counts_per_hour_of_weekday = np.zeros((7,24)) # weekdays * hours in a day
weekdays = dict()
for created_utc in DATA.created_utc.values.tolist():
    current_datetime = dt.datetime.fromtimestamp(int(created_utc))
    wd = current_datetime.weekday()
    h = current_datetime.hour
    counts_per_hour_of_weekday[wd,h]+=1
    weekdays[wd] = current_datetime.strftime('%A')
print(counts_per_hour_of_weekday.shape)
print(weekdays)
plt.figure(figsize=(14,6))
plt.imshow(counts_per_hour_of_weekday)
plt.yticks([k for k, v in weekdays.items()], [v for k, v in weekdays.items()])
plt.xticks([i for i in range(counts_per_hour_of_weekday.shape[1])], [i for i in range(counts_per_hour_of_weekday.shape[1])])
plt.ylabel('Day of the Week')
plt.xlabel('Hour of the Day')
plt.title('/r/Coronavirus submissions by hour of the day!')
plt.show()
!git clone https://github.com/COVIEWED/coviewed_web_scraping
!pip install -r coviewed_web_scraping/requirements.txt
#EXAMPLE_URL = DATA[['url']].sample(n=1).url.values.tolist()[0]
EXAMPLE_URL = 'https://edition.cnn.com/2020/03/04/health/debunking-coronavirus-myths-trnd/'
print(EXAMPLE_URL)
!echo {EXAMPLE_URL}
!rm coviewed_web_scraping/data/*.txt
!cd coviewed_web_scraping/ && python3 src/scrape.py -u={EXAMPLE_URL}
!ls coviewed_web_scraping/data/*.txt
data_path = 'coviewed_web_scraping/data/'
fname = [f for f in os.listdir(data_path) if f.endswith('.txt')][0]
with open(os.path.join(data_path, fname), 'r') as my_file:
    txt_data = my_file.readlines()
txt_data = [line.strip() for line in txt_data if line.strip()]
len(txt_data)
article_url = txt_data[0]
print(article_url)
article_published_datetime = txt_data[1]
print(article_published_datetime)
article_title = txt_data[2]
print(article_title)
article_text = "\n\n".join(txt_data[3:])
print(article_text)
print('List of claims from the article:', end='\n\n')
for row in article_text.splitlines():
    if row.strip() and 'Myth:' in row:
        print(row.strip()[len('Myth: '):])
