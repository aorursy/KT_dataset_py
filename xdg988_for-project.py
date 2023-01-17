import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import nltk

from nltk.corpus import stopwords

import string

import scipy

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
data = pd.read_csv('../input/train.csv')

y = data.deal_probability.copy()

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.20, random_state=23)



del data

gc.collect()



X_train.head()
for df in [X_train, X_test]:

    df['description'].fillna('unknowndesc', inplace=True)

    df['title'].fillna('unknowntitle', inplace=True)

    

    for col in ['description', 'title']:

        df['num_words_' + col] = df[col].apply(lambda comment: len(comment.split()))

        df['num_unique_words_' + col] = df[col].apply(lambda comment: len(set(w for w in comment.split())))



    df['words_vs_unique_title'] = df['num_unique_words_title'] / df['num_words_title'] * 100

    df['words_vs_unique_description'] = df['num_unique_words_description'] / df['num_words_description'] * 100



    df['weekday'] = pd.to_datetime(df['activation_date']).dt.weekday

    df['Day of Month'] = pd.to_datetime(df['activation_date']).dt.day

    df['city'] = df['region'] + '_' + df['city']

    df['num_desc_punct'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
# Countvectorizer

count_vectorizer_title = CountVectorizer(stop_words=stopwords.words('russian'), lowercase=True,

                                        ngram_range=(1, 2), max_features=8000)

title_counts = count_vectorizer_title.fit_transform(X_train['title'].append(X_test['title']))

train_title_counts = title_counts[:len(X_train)]

test_title_counts = title_counts[len(X_train):]



#TF-IDF

count_vectorizer_desc = TfidfVectorizer(stop_words=stopwords.words('russian'), lowercase=True,

                                        ngram_range=(1, 2), max_features=17000)

desc_counts = count_vectorizer_desc.fit_transform(X_train['description'].append(X_test['description']))



train_desc_counts = desc_counts[:len(X_train)]

test_desc_counts = desc_counts[len(X_train):]



train_title_counts.shape, train_desc_counts.shape
#target = 'deal_probability'

predictors = [

    'num_desc_punct', 'num_words_title', 'words_vs_unique_title', 'num_unique_words_title',

    'words_vs_unique_description', 'num_unique_words_description', 'num_words_description',

    'price', 'item_seq_number', 'Day of Month', 'weekday'

]

categorical = [

    'image_top_1', 'param_1', 'param_2', 'param_3', 

    'city', 'region', 'category_name', 'parent_category_name', 'user_type'

]



predictors = predictors + categorical
for feature in categorical:

    print(f'Transforming {feature}...')

    encoder = LabelEncoder()

    X_train[feature].fillna('unknown',inplace=True)

    X_test[feature].fillna('unknown',inplace=True)

    encoder.fit(X_train[feature].append(X_test[feature]).astype(str))

    

    X_train[feature] = encoder.transform(X_train[feature].astype(str))

    X_test[feature] = encoder.transform(X_test[feature].astype(str))
X_train["price"] = np.log(X_train["price"]+0.001)

X_train["price"].fillna(-1,inplace=True)

X_train["image_top_1"].fillna(-1,inplace=True)



X_test["price"] = np.log(X_test["price"]+0.001)

X_test["price"].fillna(-1,inplace=True)

X_test["image_top_1"].fillna(-1,inplace=True)
feature_names = np.hstack([

    count_vectorizer_desc.get_feature_names(),

    count_vectorizer_title.get_feature_names(),

    predictors

])

print('Number of features:', len(feature_names))
test = scipy.sparse.hstack([

    test_desc_counts,

    test_title_counts,

    X_test.loc[:, predictors]

], format='csr')



train = scipy.sparse.hstack([

    train_desc_counts,

    train_title_counts,

    X_train.loc[: , predictors]

], format='csr')
#train为训练集的数据，test为测试集的数据，对应的y分别是y_train和y_test
import lightgbm as lgb



lgbm_params = {

    'objective' : 'regression',

    'metric' : 'rmse',

    'num_leaves' : 300,

#     'max_depth': 15,

    'learning_rate' : 0.02,

    'feature_fraction' : 0.6,

    'bagging_fraction' : .8,

    'verbosity' : -1

}



lgtrain = lgb.Dataset(train, y_train,

                feature_name=list(feature_names),

                categorical_feature = categorical)

lgvalid = lgb.Dataset(test, y_test,

                feature_name=list(feature_names),

                categorical_feature = categorical)



# Go Go Go

lgb_clf = lgb.train(

    lgbm_params,

    lgtrain,

    num_boost_round=5000,

    valid_sets=[lgtrain, lgvalid],

    valid_names=['train','valid'],

    early_stopping_rounds=50,

    verbose_eval=100)

print("Model Evaluation Stage")

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lgb_clf.predict(test))))