!pip install mord==0.6
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!wc -l /kaggle/input/yelp-dataset/yelp_academic_dataset_review.json
!wc -l /kaggle/input/yelp-dataset/yelp_academic_dataset_user.json
!wc -l /kaggle/input/yelp-dataset/yelp_academic_dataset_business.json
import json
from collections import defaultdict

from tqdm import tqdm
filename = '/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json'
target_user_ids = dict()
target_item_ids = dict()
reviews = []
with open(filename, 'rt') as f:
    with tqdm(total=8021122) as pbar:
        for line in f:
            data = json.loads(line)

            if not 'date' in data:
                continue

            dt = data.get('date').split()[0] # get '2019-01-01'

            # 2019年のみに絞る。
            if dt.startswith('2019-12'):
                reviews.append({
                    key: data[key]
                    for key in ['review_id', 'user_id', 'business_id', 'stars']
                })
                target_user_ids[data['user_id']] = True
                target_item_ids[data['business_id']] = True
            pbar.update(1)
print(len(reviews))
print(json.dumps(reviews[0], indent=4))
filename = '/kaggle/input/yelp-dataset/yelp_academic_dataset_business.json'
items = list()
with open(filename, 'rt') as f:
    with tqdm(total=209393) as pbar:
        for line in f:
            data = json.loads(line)
            if data['business_id'] in target_item_ids:
                items.append(data)
            pbar.update(1)
print(len(items))
filename = '/kaggle/input/yelp-dataset/yelp_academic_dataset_user.json'
users = list()
with open(filename, 'rt') as f:
    with tqdm(total=1968703) as pbar:
        for line in f:
            data = json.loads(line)
            if data['user_id'] in target_user_ids:
                users.append(data)
            pbar.update(1)
print(len(users))
import gc

del target_user_ids
del target_item_ids
gc.collect()
items[0]
users[1]
reviews[0]
user_df = pd.DataFrame(users)
item_df = pd.DataFrame(items)
review_df = pd.DataFrame(reviews)
user_df
item_df
review_df
review_df['y'] = review_df.stars#.apply(lambda star: 1 if star >= 3 else 0)
review_df
user_df.columns
f_user_df = user_df.drop(['name', 'yelping_since', 'elite', 'friends'], axis=1)
f_user_df.shape
f_item_df = item_df.drop(['name', 'address', 'city', 'state', 'postal_code', 'attributes', 'categories', 'hours'], axis=1)
f_item_df.shape
f_review_df = review_df.drop(['stars'], axis=1)
f_review_df.shape
merge_df = pd.merge(f_review_df, f_user_df, on='user_id')
merge_df = pd.merge(merge_df, f_item_df, on='business_id')
merge_df = merge_df.drop(['review_id', 'user_id', 'business_id'], axis=1)
merge_df
X = merge_df.drop(['y'], axis=1).values
y = merge_df.y.values.astype(np.int64)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mord import LogisticAT
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape)
print(X_test.shape)
model = LogisticRegression(max_iter=1000, verbose=1, n_jobs=-1)
at_model = LogisticAT()
model.fit(X_train, y_train)
at_model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, model.predict(X_test))
accuracy_score(y_test, at_model.predict(X_test))
cnt = 0
for ac, pr, at_pr in zip(y_test, model.predict(X_test), at_model.predict(X_test)):
    print(f'(actual, predict, predictAt)=({ac}, {pr}, {at_pr})')
    if cnt == 20:
        break
    cnt+=1
pd.Series(y_train).value_counts().sort_index(ascending=False)
pd.Series(model.predict(X_test)).value_counts().sort_index(ascending=False)
at_model.predict_proba(X_test)
model.predict_proba(X_test)
