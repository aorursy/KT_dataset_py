import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
%matplotlib inline

from IPython.display import display

import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('/kaggle/input/amazon-data-science-book-reviews/reviews.csv')
data.head()
data.shape
data.loc[0].comment
data.drop(['book_url'], axis=1, inplace=True)
data.stars.hist()
data.stars.value_counts()
data[data.comment.apply(len)<50].stars.value_counts()
data = data.drop(data[data.comment.apply(len)<50][data.stars==5.0].index)
data.stars.value_counts()
import nltk
# from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.comment, data.stars, test_size=0.3, random_state=37)
from sklearn import metrics  # подгружаем метрики
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def dataframe_metrics(y_test,y_pred):
    stats = [
       metrics.mean_absolute_error(y_test, y_pred),
       np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
       metrics.r2_score(y_test, y_pred),
       mean_absolute_percentage_error(y_test, y_pred)
    ]
    return stats
measured_metrics = pd.DataFrame({"error_type":["MAE", "RMSE", "R2", "MAPE"]})
y_mean = np.median(y_train)
y_pred_naive = np.ones(len(y_test)) * y_mean
measured_metrics["naive"] = dataframe_metrics(y_test, y_pred_naive)
measured_metrics
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english',
                             norm=None,
                             max_features=1500,
                             min_df=0,
                             max_df=0.2,
                             ngram_range=(1,2))

features_train = vectorizer.fit_transform(X_train).todense()
features_test = vectorizer.transform(X_test).todense()

train_matrix = pd.DataFrame(
    features_train, 
    columns=vectorizer.get_feature_names()
)

test_matrix = pd.DataFrame(
    features_test, 
    columns=vectorizer.get_feature_names()
)
len(vectorizer.vocabulary_)
features_train.shape, features_test.shape
train_matrix.head(10)
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(cv=10, n_jobs=-1)
lasso_cv.fit(train_matrix, y_train)

y_pred_lasso = lasso_cv.predict(test_matrix)

measured_metrics["tf-idf"] = dataframe_metrics(y_test, y_pred_lasso)
measured_metrics
featureImportance = pd.DataFrame({"feature": train_matrix.columns[abs(lasso_cv.coef_)>0.04], 
                                  "importance": lasso_cv.coef_[abs(lasso_cv.coef_)>0.04]})

featureImportance.set_index('feature', inplace=True)
featureImportance.sort_values(["importance"], ascending=False, inplace=True)
featureImportance["importance"].plot.bar(figsize=(25,15));