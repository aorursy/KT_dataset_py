import pandas as pd

import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))

df = pd.read_csv('../input/train.tsv',sep="\t")
df.head()
df.set_index('train_id',inplace=True)
# quick check on missing data

df_na = (df.isnull().sum() / len(df)) * 100

df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :df_na})

missing_data.head(20)
len(df[df['price'] == 0])
df = df[df['price']>0]
# split target and features

y = df['price']

df = df.drop('price',axis=1)
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

from scipy.stats import norm



sns.distplot(y , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(y)



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')

plt.show()
import numpy as np

y_log = np.log1p(y)
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

from scipy.stats import norm



sns.distplot(y_log , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(y_log)



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')

plt.show()
df.dtypes
# replace missing values with 'missing'

df['brand_name'] = df['brand_name'].fillna('missing')

df['category_name'] = df['category_name'].fillna('missing')

df['item_description'] = df['item_description'].fillna('missing')
df.head()
# changing the column types for categorical features

df['category_name'] = df['category_name'].astype('category')

df['brand_name'] = df['brand_name'].astype('category')

df['item_condition_id'] = df['item_condition_id'].astype('category')
# clean up text based features before tf-idf 

def clean_text(col):

    # remove non alpha characters

    col = col.str.replace("[\W]", " ") #a-zA-Z1234567890

    # all lowercase

    col = col.apply(lambda x: x.lower())

    return col



df['name']=clean_text(df['name'])

df['category_name']=clean_text(df['category_name'])

df['item_description']=clean_text(df['item_description'])
df.head()
# create feature matrix for name and category_name

from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(min_df=10,max_df=0.1, stop_words='english')

X_name = cv.fit_transform(df['name'])

cv = CountVectorizer()

X_category = cv.fit_transform(df['category_name'])
X_name.shape
X_category.shape
# Feature matrix for item description

cv = CountVectorizer(min_df=10,max_df=0.1, stop_words='english')

X_item_description = cv.fit_transform(df['item_description'])
X_item_description.shape
# feature matrix for brand

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer(sparse_output=True)

X_brand = lb.fit_transform(df['brand_name'])

X_brand.shape
# feature matrix for item condition and shipping

from scipy.sparse import csr_matrix

X_condition_shipping = csr_matrix(pd.get_dummies(df[['item_condition_id', 'shipping']], sparse=True).values)
X_condition_shipping.shape
# create the complete feature matrix

from scipy.sparse import hstack



X_all = hstack((X_brand, X_category, X_name, X_item_description, X_condition_shipping)).tocsr()
X_all.shape
# reduce the feature columns by removing all features with a document frequency smaller than 1

mask = np.array(np.clip(X_all.getnnz(axis=0) - 1, 0, 1), dtype=bool)

X_all = X_all[:, mask]
X_all.shape
# split into test and train samples

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_all, y_log, random_state=42, train_size=0.1, test_size=0.02)
X_train.shape
X_test.shape
from sklearn.model_selection import KFold, cross_val_score

def score_model(model):

    kf = KFold(3, shuffle=True, random_state=42).get_n_splits(X_train)

    model_score = np.mean(cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf, n_jobs=-1))

    return((type(model).__name__,model_score))
# get a baseline for a few regression models



import time

from sklearn.linear_model import Ridge, Lasso

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor



model_scores = pd.DataFrame(columns=['model','NMSE'])

reg_model = [Ridge(),Lasso(), GradientBoostingRegressor(),XGBRegressor()]

for model in reg_model:

    start = time.time()

    sc = score_model(model)

    total = time.time() - start

    print("done with {}, ({}s)".format(sc[0],total))

    model_scores = model_scores.append({'model':sc[0],'NMSE':sc[1]},ignore_index=True)    



# print results

model_scores.sort_values('NMSE',ascending=False)