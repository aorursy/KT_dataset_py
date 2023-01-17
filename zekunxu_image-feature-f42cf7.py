# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from PIL import Image, ImageDraw, ImageFilter

import matplotlib.pyplot as plt

%matplotlib inline

import nltk

import glob
plt.rcParams['figure.figsize'] = (10.0, 10.0)

images = sorted(glob.glob('../input/images_sample/**/**.jpg'))

for im in images:

    im = Image.open(im)

    h, w = im.size

    qu = im.quantize(colors=8, kmeans=4)

    crgb = qu.convert('RGB')

    col_rank = sorted(crgb.getcolors(h*w), reverse=True)

    print(col_rank) #legend

    draw = ImageDraw.Draw(im)

    i = 0

    for cnt, rgb in col_rank:

        draw.rectangle([(10, i*40+10),(40, i*40+30)], fill=(rgb[0],rgb[1],rgb[2]), outline=(0,0,0))

        draw.text((10, i*40+30), str(cnt), fill=(0,0,0))

        i += 1

    del draw

    plt.imshow(im); plt.axis('off')

    break
from PIL import ImageStat

for im in images:

    img = Image.open(im)

    stats = ImageStat.Stat(img, mask=None)

    print(stats.extrema)

    print(stats.count)

    print(stats.sum)

    print(stats.sum2)

    print(stats.mean)

    print(stats.median)

    print(stats.rms)

    print(stats.var)

    print(stats.stddev)

    plt.imshow(img); plt.axis('off')

    break
import numpy as np

import imagehash, hashlib

import random



images = glob.glob('../input/images_sample/6812098/**.jpg') #just comparing two folders for demo

images += glob.glob('../input/images_sample/6812035/**.jpg')



for im in range(100):

    im1 = random.choice(images)

    im2 = random.choice(images)

    h1 = imagehash.dhash(Image.open(im1))

    h2 = imagehash.dhash(Image.open(im2))

    feature = h1 - h2

    if feature < 7 and im1 != im2:

        print(feature, im1, im2)

        imgx = np.concatenate((Image.open(im1).resize((400, 400), Image.ANTIALIAS), Image.open(im2).resize((400, 400), Image.ANTIALIAS)), axis=1)

        plt.imshow(imgx); plt.axis('off')

        break
import time; start_time = time.time()

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import model_selection, preprocessing

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import FeatureUnion

from sklearn.metrics import log_loss

from sklearn import pipeline

import pandas as pd

import numpy as np

from nltk.stem.porter import *

stemmer = PorterStemmer()

from bs4 import BeautifulSoup

import random; random.seed(7)

import xgboost as xgb

import datetime as dt



train = pd.read_json(open("../input/train.json", "r"))[:10000] #limit

y = train.interest_level.values

n = len(train)



test = pd.read_json(open("../input/test.json", "r"))[:10000] #limit

listing_id = test.listing_id.values



col = [x for x in train.columns if x not in ['listing_id','interest_level','street_address']]

print(col)

print(len(train),len(test))



def str_stem(s): 

    if isinstance(s, str):

        s = s.lower()

        s = s.replace("  "," ")

        b = BeautifulSoup(s, "lxml")

        s = b.get_text(" ").strip()

        s = (" ").join([z for z in s.split(" ")])

        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])

        s = s.lower().strip()

        return s

    else:

        return ""



class cust_regression_vals(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):

        return self

    def transform(self, df):

        d_col_drops=['xdescription', 'ydescription']

        df = df.drop(d_col_drops, axis=1).values

        return df



class cust_txt_col(BaseEstimator, TransformerMixin):

    def __init__(self, key):

        self.key = key

    def fit(self, x, y=None):

        return self

    def transform(self, data_dict):

        return data_dict[self.key].apply(str)

    

df_all = pd.concat((train[col], test[col]), axis=0, ignore_index=True)

train = []

test = []



df_all['photos'] = df_all.photos.apply(len)



df_all["price_be"] = df_all["price"]/df_all["bedrooms"]

df_all["price_ba"] = df_all["price"]/df_all["bathrooms"]



df_all["created"] = pd.to_datetime(df_all["created"])

df_all["created_year"] = df_all["created"].dt.year

df_all["created_month"] = df_all["created"].dt.month

df_all["created_day"] = df_all["created"].dt.day

df_all['created_hour'] = df_all["created"].dt.hour

df_all['created_weekday'] = df_all['created'].dt.weekday

df_all['created_week'] = df_all['created'].dt.week

df_all['created_quarter'] = df_all['created'].dt.quarter

df_all['created_weekend'] = ((df_all['created_weekday'] == 5) & (df_all['created_weekday'] == 6))

df_all['created_wd'] = ((df_all['created_weekday'] != 5) & (df_all['created_weekday'] != 6))

df_all['created'] = df_all['created'].map(lambda x: float((x - dt.datetime(1899, 12, 30)).days) + (float((x - dt.datetime(1899, 12, 30)).seconds) / 86400))



df_all['x5'] = df_all['latitude'].map(lambda x : round(x,5))

df_all['y5'] = df_all['longitude'].map(lambda x : round(x,5))

df_all['x4'] = df_all['latitude'].map(lambda x : round(x,4))

df_all['y4'] = df_all['longitude'].map(lambda x : round(x,4))

df_all['x3'] = df_all['latitude'].map(lambda x : round(x,3))

df_all['y3'] = df_all['longitude'].map(lambda x : round(x,3))

df_all['x2'] = df_all['latitude'].map(lambda x : round(x,2))

df_all['y2'] = df_all['longitude'].map(lambda x : round(x,2))



dummies = df_all['features'].str.join(sep=',').str.lower().str.get_dummies(sep=',')

df_all = pd.concat([df_all, dummies], axis=1)

dummies = []

df_all['features'] = df_all.features.apply(len)



cat = ['building_id',  'description', 'display_address', 'manager_id']

lbl = preprocessing.LabelEncoder()

for c in cat:

    if c in ['description']:

        df_all['x'+c] = df_all[c].map(lambda x:str_stem(x))

        df_all['y'+c] = df_all[c].values

    df_all['words_of_'+c] = df_all[c].map(lambda x:len(x.strip().split(' ')))

    df_all['len_of_'+c] = df_all[c].map(lambda x:len(x.strip()))

    df_all[c] = lbl.fit_transform(list(df_all[c].values))

    print(c, len(lbl.classes_))



train = df_all.iloc[:n]

test = df_all.iloc[n:]

#df_all = []



tfidf = TfidfVectorizer(stop_words ='english', max_df=0.9)

tsvd = TruncatedSVD(n_components=25, random_state = 7)

clf = pipeline.Pipeline([

        ('union', FeatureUnion(

                    transformer_list = [

                        ('cst',  cust_regression_vals()),

                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='xdescription')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),

                        ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='ydescription')), ('tfidf2', tfidf), ('tsvd2', tsvd)]))

                        ],

                    transformer_weights = {

                        'cst': 1.0,

                        'txt1': 1.0,

                        'txt2': 1.0

                        },

                n_jobs = -1

                ))])



y_val = lbl.fit_transform(y)

xtrain = pd.DataFrame(clf.fit_transform(train)).apply(pd.to_numeric)

xtrain = xgb.DMatrix(xtrain.values, y_val)

xtest = pd.DataFrame(clf.transform(test)).apply(pd.to_numeric)

xtest = xgb.DMatrix(xtest.values)



param = {}

param['objective'] = 'multi:softprob'

param['eta'] = 0.1

#param['max_depth'] = 4

param['silent'] = True

param['num_class'] = 3

param['eval_metric'] = "mlogloss"

param['min_child_weight'] = 1

param['subsample'] = 0.7

param['colsample_bytree'] = 0.7

param['seed'] = 7

plst = list(param.items())

nfolds = 5

nrounds = 1000



model = xgb.cv(plst, xtrain, nrounds, nfolds, early_stopping_rounds=20, verbose_eval=25)

best_rounds = np.argmin(model['test-mlogloss-mean'])

model = xgb.train(plst, xtrain, best_rounds)

print(log_loss(y_val, model.predict(xtrain)))

preds = model.predict(xtest)

out_df = pd.DataFrame(preds)

out_df.columns = lbl.inverse_transform(out_df.columns)

out_df["listing_id"] = listing_id

#out_df = out_df[-1:] + out_df[:-1]

out_df = out_df[['listing_id', 'high', 'low', 'medium']]

out_df.to_csv("z09submission01.csv", index=False)

print('Done...',(time.time()-start_time)/60)
null