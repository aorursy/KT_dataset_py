import pandas as pd

import numpy as np

import random

import sys

import os

!pip install tldextract -q

import tldextract

import warnings

import regex as re

import eli5

from typing import *



from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from sklearn.compose import ColumnTransformer

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import FunctionTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn import svm

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC



import matplotlib.pyplot as plt

import seaborn as sns 

from urllib.parse import urlparse

from nltk.tokenize import RegexpTokenizer



warnings.filterwarnings("ignore")
url_data = pd.read_csv('/kaggle/input/phishing-site-urls/phishing_site_urls.csv')

#pd.set_option('display.max_colwidth', -1)

url_data.sample(10)
url_data = url_data.rename(columns={"URL": "url", "Label": "label"})
def parse_url(url: str) -> Optional[Dict[str, str]]:

    try:

        no_scheme = not url.startswith('https://') and not url.startswith('http://')

        if no_scheme:

            parsed_url = urlparse(f"http://{url}")

            return {

                "scheme": None, # not established a value for this

                "netloc": parsed_url.netloc,

                "path": parsed_url.path,

                "params": parsed_url.params,

                "query": parsed_url.query,

                "fragment": parsed_url.fragment,

            }

        else:

            parsed_url = urlparse(url)

            return {

                "scheme": parsed_url.scheme,

                "netloc": parsed_url.netloc,

                "path": parsed_url.path,

                "params": parsed_url.params,

                "query": parsed_url.query,

                "fragment": parsed_url.fragment,

            }

    except:

        return None
url_data["parsed_url"] = url_data.url.apply(parse_url)

url_data
url_data = pd.concat([

    url_data.drop(['parsed_url'], axis=1),

    url_data['parsed_url'].apply(pd.Series)

], axis=1)

url_data
url_data = url_data[~url_data.netloc.isnull()]

url_data
url_data["length"] = url_data.url.str.len()
url_data["tld"] = url_data.netloc.apply(lambda nl: tldextract.extract(nl).suffix)

url_data['tld'] = url_data['tld'].replace('','None')
url_data["is_ip"] = url_data.netloc.str.fullmatch(r"\d+\.\d+\.\d+\.\d+")
url_data['domain_hyphens'] = url_data.netloc.str.count('-')

url_data['domain_underscores'] = url_data.netloc.str.count('_')

url_data['path_hyphens'] = url_data.path.str.count('-')

url_data['path_underscores'] = url_data.path.str.count('_')

url_data['slashes'] = url_data.path.str.count('/')
url_data['full_stops'] = url_data.path.str.count('.')
def get_num_subdomains(netloc: str) -> int:

    subdomain = tldextract.extract(netloc).subdomain 

    if subdomain == "":

        return 0

    return subdomain.count('.') + 1



url_data['num_subdomains'] = url_data['netloc'].apply(lambda net: get_num_subdomains(net))
tokenizer = RegexpTokenizer(r'[A-Za-z]+')

def tokenize_domain(netloc: str) -> str:

    split_domain = tldextract.extract(netloc)

    no_tld = str(split_domain.subdomain +'.'+ split_domain.domain)

    return " ".join(map(str,tokenizer.tokenize(no_tld)))

         

url_data['domain_tokens'] = url_data['netloc'].apply(lambda net: tokenize_domain(net))
url_data['path_tokens'] = url_data['path'].apply(lambda path: " ".join(map(str,tokenizer.tokenize(path))))
url_data_y = url_data['label']

url_data.drop('label', axis=1, inplace=True)

url_data.drop('url', axis=1, inplace=True)

url_data.drop('scheme', axis=1, inplace=True)

url_data.drop('netloc', axis=1, inplace=True)

url_data.drop('path', axis=1, inplace=True)

url_data.drop('params', axis=1, inplace=True)

url_data.drop('query', axis=1, inplace=True)

url_data.drop('fragment', axis=1, inplace=True)

url_data
class Converter(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):

        return self



    def transform(self, data_frame):

        return data_frame.values.ravel()
X_train, X_test, y_train, y_test = train_test_split(url_data, url_data_y, test_size=0.2)
numeric_features = ['length', 'domain_hyphens', 'domain_underscores', 'path_hyphens', 'path_underscores', 'slashes', 'full_stops', 'num_subdomains']

numeric_transformer = Pipeline(steps=[

    ('scaler', MinMaxScaler())])
categorical_features = ['tld', 'is_ip']

categorical_transformer = Pipeline(steps=[

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
vectorizer_features = ['domain_tokens','path_tokens']

vectorizer_transformer = Pipeline(steps=[

    ('con', Converter()),

    ('tf', TfidfVectorizer())])
preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features),

        ('domvec', vectorizer_transformer, ['domain_tokens']),

        ('pathvec', vectorizer_transformer, ['path_tokens'])

    ])



svc_clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', LinearSVC())])



log_clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', LogisticRegression())])



nb_clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', MultinomialNB())])
svc_clf.fit(X_train, y_train)

log_clf.fit(X_train, y_train)

nb_clf.fit(X_train, y_train)
def results(name: str, model: BaseEstimator) -> None:

    preds = model.predict(X_test)



    print(name + " score: %.3f" % model.score(X_test, y_test))

    print(classification_report(y_test, preds))

    labels = ['Good', 'Bad']



    conf_matrix = confusion_matrix(y_test, preds)



    font = {'family' : 'normal',

            'size'   : 14}



    plt.rc('font', **font)

    plt.figure(figsize= (10,6))

    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", cmap='Blues')

    plt.title("Confusion Matrix for " + name)

    plt.ylabel('True Class')

    plt.xlabel('Predicted Class')
results("SVC" , svc_clf)

results("Logistic Regression" , log_clf)

results("Naive Bayes" , nb_clf)
onehot_columns = list(svc_clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names(input_features=categorical_features))

domvect_columns = list(svc_clf.named_steps['preprocessor'].named_transformers_['domvec'].named_steps['tf'].get_feature_names())

pathvect_columns = list(svc_clf.named_steps['preprocessor'].named_transformers_['pathvec'].named_steps['tf'].get_feature_names())

numeric_features_list = list(numeric_features)

numeric_features_list.extend(onehot_columns)

numeric_features_list.extend(domvect_columns)

numeric_features_list.extend(pathvect_columns)

eli5.explain_weights(svc_clf.named_steps['classifier'], top=20, feature_names=numeric_features_list)