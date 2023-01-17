# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import squarify

import matplotlib.dates as dates

from datetime import datetime



%matplotlib inline
df = pd.read_csv('/kaggle/input/ecommerce-behavior-data-from-multi-category-store/2019-Nov.csv')
df.head()
df.info()
df.shape
df.columns
visitor = df['user_id'].nunique()

print ("visitors: {}".format(visitor))
d = df.loc[:,['event_time','user_id']]
d['event_time'] = d['event_time'].apply(lambda s: str(s)[0:10])



visitor_by_date = d.drop_duplicates().groupby(['event_time'])['user_id'].agg(['count']).sort_values(by=['event_time'], ascending=True)
x = pd.Series(visitor_by_date.index.values).apply(lambda s: datetime.strptime(s, '%Y-%m-%d').date())

y = visitor_by_date['count']

plt.rcParams['figure.figsize'] = (20,8)



plt.plot(x,y)

plt.show()
top_category_n = 30

top_category = df.loc[:,'category_code'].value_counts()[:top_category_n].sort_values(ascending=False)

squarify.plot(sizes=top_category, label=top_category.index.array, color=["red","cyan","green","orange","blue","grey"], alpha=.7  )

plt.axis('off')

plt.show()
labels = ['view', 'cart','purchase']

size = df['event_type'].value_counts()

colors = ['yellowgreen', 'lightskyblue','lightcoral']

explode = [0, 0.1,0.1]



plt.rcParams['figure.figsize'] = (8, 8)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('Event_Type', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
purchase = df.loc[df['event_type'] == 'purchase']

purchase = purchase.dropna(axis='rows')

purchase.head()
top_sellers = purchase.groupby('brand')['brand'].agg(['count']).sort_values('count', ascending=False)

top_sellers.head(20)
df_targets = df.loc[df["event_type"].isin(["cart","purchase"])].drop_duplicates(subset=['event_type', 'product_id','price', 'user_id','user_session'])

df_targets["is_purchased"] = np.where(df_targets["event_type"]=="purchase",1,0)

df_targets["is_purchased"] = df_targets.groupby(["user_session","product_id"])["is_purchased"].transform("max")

df_targets = df_targets.loc[df_targets["event_type"]=="cart"].drop_duplicates(["user_session","product_id","is_purchased"])

df_targets['event_weekday'] = df_targets['event_time'].apply(lambda s: str(datetime.strptime(str(s)[0:10], "%Y-%m-%d").weekday()))

df_targets.dropna(how='any', inplace=True)

df_targets["category_code_level1"] = df_targets["category_code"].str.split(".",expand=True)[0].astype('category')

df_targets["category_code_level2"] = df_targets["category_code"].str.split(".",expand=True)[1].astype('category')
cart_purchase_users = df.loc[df["event_type"].isin(["cart","purchase"])].drop_duplicates(subset=['user_id'])

cart_purchase_users.dropna(how='any', inplace=True)

cart_purchase_users_all_activity = df.loc[df['user_id'].isin(cart_purchase_users['user_id'])]


activity_in_session = cart_purchase_users_all_activity.groupby(['user_session'])['event_type'].count().reset_index()

activity_in_session = activity_in_session.rename(columns={"event_type": "activity_count"})
del d # free memory


df_targets = df_targets.merge(activity_in_session, on='user_session', how='left')

df_targets['activity_count'] = df_targets['activity_count'].fillna(0)

df_targets.head()
df_targets.to_csv('training_data.csv')
df_targets.info()
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from xgboost import plot_importance

from sklearn.utils import resample

from sklearn import metrics
is_purcahase_set = df_targets[df_targets['is_purchased']== 1]

is_purcahase_set.shape[0]
not_purcahase_set = df_targets[df_targets['is_purchased']== 0]

not_purcahase_set.shape[0]
n_samples = 500000

is_purchase_downsampled = resample(is_purcahase_set,

                                replace = False, 

                                n_samples = n_samples,

                                random_state = 27)

not_purcahase_set_downsampled = resample(not_purcahase_set,

                                replace = False,

                                n_samples = n_samples,

                                random_state = 27)
downsampled = pd.concat([is_purchase_downsampled, not_purcahase_set_downsampled])

downsampled['is_purchased'].value_counts()
features = downsampled.loc[:,['brand', 'price', 'event_weekday', 'category_code_level1', 'category_code_level2', 'activity_count']]
features.loc[:,'brand'] = LabelEncoder().fit_transform(downsampled.loc[:,'brand'].copy())

features.loc[:,'event_weekday'] = LabelEncoder().fit_transform(downsampled.loc[:,'event_weekday'].copy())

features.loc[:,'category_code_level1'] = LabelEncoder().fit_transform(downsampled.loc[:,'category_code_level1'].copy())

features.loc[:,'category_code_level2'] = LabelEncoder().fit_transform(downsampled.loc[:,'category_code_level2'].copy())



is_purchased = LabelEncoder().fit_transform(downsampled['is_purchased'])

features.head()
print(list(features.columns))
X_train, X_test, y_train, y_test = train_test_split(features, 

                                                    is_purchased, 

                                                    test_size = 0.3, 

                                                    random_state = 0)
from xgboost import XGBClassifier

model = XGBClassifier(learning_rate=0.1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))

print("fbeta:",metrics.fbeta_score(y_test, y_pred, average='weighted', beta=0.5))
plot_importance(model, max_num_features=10, importance_type ='gain')

plt.rcParams['figure.figsize'] = (40,10)

plt.show()