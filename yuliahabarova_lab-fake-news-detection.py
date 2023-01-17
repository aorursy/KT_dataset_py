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
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from stop_words import get_stop_words

import warnings
warnings.filterwarnings('ignore')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/textdb3/fake_or_real_news.csv', index_col=0)
df.head()
df['target'] = df['label'].map({'FAKE': 0, 'REAL' : 1})
data = df.drop('label', axis = 1)
english_stops  = set(get_stop_words('english'))
bow = CountVectorizer(stop_words=english_stops, min_df=5)
tfidf = TfidfVectorizer(stop_words=english_stops, min_df=5)

X_title_bow = bow.fit_transform(df['title'])

X_train, X_valid, y_train, y_valid = train_test_split(X_title_bow, df['target'], test_size=0.25, random_state=1)

elastic = ElasticNet(l1_ratio=0, random_state=1)

elastic.fit(X_train, y_train)   

y_pred = elastic.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
linear = LinearRegression()
linear.fit(X_train, y_train)   

y_pred = linear.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
X_title_tfidf = tfidf.fit_transform(df['title'])

X_train, X_valid, y_train, y_valid = train_test_split(X_title_tfidf, df['target'], test_size=0.25, random_state=1)

elastic = ElasticNet(l1_ratio=0, random_state=1)

elastic.fit(X_train, y_train)   

y_pred = elastic.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
linear = LinearRegression()
linear.fit(X_train, y_train)   

y_pred = linear.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
X_text_bow = bow.fit_transform(df['text'])

X_train, X_valid, y_train, y_valid = train_test_split(X_text_bow, df['target'], test_size=0.25, random_state=1)

elastic = ElasticNet(l1_ratio=0, random_state=1)

elastic.fit(X_train, y_train)   

y_pred = elastic.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
linear = LinearRegression()
linear.fit(X_train, y_train)   

y_pred = linear.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
X_text_tfidf = tfidf.fit_transform(df['text'])

X_train, X_valid, y_train, y_valid = train_test_split(X_text_tfidf, df['target'], test_size=0.25, random_state=1)

elastic = ElasticNet(l1_ratio=0, random_state=1)

elastic.fit(X_train, y_train)   

y_pred = elastic.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
linear = LinearRegression()
linear.fit(X_train, y_train)   

y_pred = linear.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
title_tfidf = tfidf.fit_transform(df['title']) 
text_tfidf = tfidf.fit_transform(df['text'])

X = np.hstack([title_tfidf, text_tfidf])

X_train, X_valid, y_train, y_valid = train_test_split(X, df['target'], test_size=0.25, random_state=1)

elastic = ElasticNet(l1_ratio=0, random_state=1)

elastic.fit(X_train, y_train)   

y_pred = elastic.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))

linear = LinearRegression()
linear.fit(X_train, y_train)   

y_pred = linear.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(stop_words=english_stops, ngram_range=(1, 2), min_df=5)
X_text_tfidf = tfidf.fit_transform(df['text'])

X_train, X_valid, y_train, y_valid = train_test_split(X_text_tfidf, df['target'], test_size=0.25, random_state=1)

elastic = ElasticNet(l1_ratio=0, random_state=1)

elastic.fit(X_train, y_train)   

y_pred = elastic.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
linear = LinearRegression()
linear.fit(X_train, y_train)   

y_pred = linear.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
X_title_tfidf = tfidf.fit_transform(df['title'])

X_train, X_valid, y_train, y_valid = train_test_split(X_title_tfidf, df['target'], test_size=0.25, random_state=1)

elastic = ElasticNet(l1_ratio=0, random_state=1)

elastic.fit(X_train, y_train)   

y_pred = elastic.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
linear = LinearRegression()
linear.fit(X_train, y_train)   

y_pred = linear.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
tfidf = TfidfVectorizer(stop_words=english_stops, ngram_range=(2, 2), min_df=5)
X_text_tfidf = tfidf.fit_transform(df['text'])

X_train, X_valid, y_train, y_valid = train_test_split(X_text_tfidf, df['target'], test_size=0.25, random_state=1)

elastic = ElasticNet(l1_ratio=0, random_state=1)

elastic.fit(X_train, y_train)   

y_pred = elastic.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
linear = LinearRegression()
linear.fit(X_train, y_train)   

y_pred = linear.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
X_title_tfidf = tfidf.fit_transform(df['title'])

X_train, X_valid, y_train, y_valid = train_test_split(X_title_tfidf, df['target'], test_size=0.25, random_state=1)

elastic = ElasticNet(l1_ratio=0, random_state=1)

elastic.fit(X_train, y_train)   

y_pred = elastic.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
linear = LinearRegression()
linear.fit(X_train, y_train)   

y_pred = linear.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
enet_params = {'alpha': np.logspace(-4, -3, 5)}

enet_grid = GridSearchCV(ElasticNet(), enet_params, cv=5, scoring='neg_mean_squared_error')
enet_grid.fit(X_title_tfidf, df['target'])
enet_grid.best_score_ , enet_grid.best_params_
enet_params = {'l1_ratio': np.linspace(0.1, 0.9, 9)}

enet_grid = GridSearchCV(ElasticNet(alpha=0.0001), 
                         enet_params, cv=5, scoring='neg_mean_squared_error')
enet_grid.fit(X_title_tfidf, df['target'])
enet_grid.best_score_ , enet_grid.best_params_
X_title_tfidf = tfidf.fit_transform(df['title'])

X_train, X_valid, y_train, y_valid = train_test_split(X_title_tfidf, df['target'], test_size=0.25, random_state=1)

elastic = ElasticNet(l1_ratio=0.30000000000000004, alpha=0.0001, random_state=1)

elastic.fit(X_train, y_train)   

y_pred = elastic.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))
enet_params = {'alpha': np.logspace(-4, -3, 5)}

enet_grid = GridSearchCV(ElasticNet(), enet_params, cv=5, scoring='neg_mean_squared_error')
enet_grid.fit(X_text_tfidf, df['target'])
enet_grid.best_score_ , enet_grid.best_params_
enet_params = {'l1_ratio': np.linspace(0.1, 0.9, 9)}

enet_grid = GridSearchCV(ElasticNet(alpha=0.0001), 
                         enet_params, cv=5, scoring='neg_mean_squared_error')
enet_grid.fit(X_text_tfidf, df['target'])
enet_grid.best_score_ , enet_grid.best_params_
X_text_tfidf = tfidf.fit_transform(df['text'])

X_train, X_valid, y_train, y_valid = train_test_split(X_text_tfidf, df['target'], test_size=0.25, random_state=1)

elastic = ElasticNet(l1_ratio=0.30000000000000004, alpha=0.0001, random_state=1)

elastic.fit(X_train, y_train)   

y_pred = elastic.predict(X_valid)
    
print('MSE:', mean_squared_error(y_valid, y_pred))
print('MAE:', mean_absolute_error(y_valid, y_pred))
print('MedAE:', median_absolute_error(y_valid, y_pred))
print('R2:', r2_score(y_valid, y_pred))