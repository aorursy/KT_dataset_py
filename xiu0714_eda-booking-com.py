#importing necessery libraries for future analysis of the dataset

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import seaborn as sns

import json



! cp /kaggle/input/bookingcom-hotel-listings/marketing_sample_for_booking_com-hotel_booking_com__20200101_20200331.ldjson booking.json

json_file = 'booking.json'

with open(json_file, 'r', encoding='utf-8') as f:  # read in the file

    list_of_rows = [json.loads(row) for row in f.readlines()]  # use a list comprehesion to convert each row from str to dict



# convert to a dataframe

df = pd.json_normalize(list_of_rows)

# df['pageurl'] = df.pageurl.apply(lambda pu: str(pu).split('/')[-1])

df.info()

df['hotel_type'].fillna(df['hotel_type'].mode()[0], inplace=True)

df.dropna(subset=['area'], inplace=True)

df.isna().sum()
df.city.value_counts()
from wordcloud import WordCloud

import os

from PIL import Image

import urllib



# Control the font for our wordcloud

if not os.path.exists('Comfortaa-Regular.ttf'):

    urllib.request.urlretrieve('http://git.io/JTqLk', 'Comfortaa-Regular.ttf')



# Get a simple India map

if not os.path.exists('india.jpg'):

    urllib.request.urlretrieve('http://git.io/JTqLa', 'india.jpg')



    

text = ' '.join(' '.join(t) for t in df.amenities.str.split('|'))

mask = np.array(Image.open('india.jpg'))

wc = WordCloud(max_words=200, background_color='white', 

              font_path='./Comfortaa-Regular.ttf', mask=mask,

              width=mask.shape[1], height=mask.shape[0]).generate(text)



plt.figure(figsize=(24,12))

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.show()
plt.figure(figsize=(16, 8))

for f in ['average_rating', 'cleanliness', 'facilities', 'location', 'staff', 'wifi', 'comfort']:

    dfcopy = df.dropna(subset=[f])

    sns.kdeplot(dfcopy[f], shade=True, alpha=.5)