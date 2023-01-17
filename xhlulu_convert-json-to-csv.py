import json

import os



import numpy as np

import pandas as pd

from tqdm import tqdm
data = {'stars': [], 'text': []}



with open('../input/yelp_academic_dataset_review.json') as f:

    for line in tqdm(f):

        review = json.loads(line)

        data['stars'].append(review['stars'])

        data['text'].append(review['text'])
df = pd.DataFrame(data)



print(df.shape)

df.head()
df['stars'] = df['stars'].astype('category')

df['text'] = df['text'].astype(str)
df.to_csv('yelp_reviews.csv', index=False)