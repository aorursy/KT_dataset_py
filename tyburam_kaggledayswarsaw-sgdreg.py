import numpy as np
import random

seed = 2018
PYTHONHASHSEED = seed
random.seed(seed)
np.random.seed(seed)
import os
os.environ['OMP_NUM_THREADS'] = '4'
import pandas as pd
from sklearn.linear_model import SGDRegressor
import gc
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
data = pd.read_csv("../input/kaggledays-warsaw/train.csv", sep="\t", index_col='id')
print(data.head())
print(data.info())
gc.collect()
col_types = {"question_id":"str", "subreddit":"str",
            "question_text": "str", "question_score":"int64",
            "answer_text":"str", "answer_score":"int64"}
train_cols = ['question_id', 'subreddit', 'question_utc', 'question_text', 'question_score', 'answer_utc', 'answer_text']
test_col = 'answer_score'

def load_data(filename="../input/kaggledays-warsaw/train.csv"):
    data = pd.read_csv(filename, sep="\t", index_col='id', dtype=col_types)
    msg = "Reading the data ({} rows). Columns: {}"
    print(msg.format(len(data), data.columns))
    # Select the columns (feel free to select more)
    X = data.loc[:, train_cols]
    try:
        y = data.loc[:, test_col]
    except KeyError: # There are no answers in the test file
        return X, None
    return X, y
def feature_engineering(df):
    df['question_len'] = df.apply(lambda row: len(row['question_text']), axis=1) #length of an question
    df['answer_len'] = df.apply(lambda row: len(row['answer_text']), axis=1) #length of an answer
    
    #links in content
    url_regex = 'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    img_regex = 'https?:[^)''"]+\.(?:jpg|jpeg|gif|png)'

    df["answer_imgs"] = df["answer_text"].apply(lambda x: len(re.findall(img_regex, x))) #number of imgs in answer
    df["answer_links"] = df["answer_text"].apply(lambda x: len(re.findall(url_regex, x))) #number of links  that are not imgs
    df["answer_links"] = df["answer_links"] - df["answer_imgs"]
    df.answer_imgs = df.answer_imgs.apply(lambda x: 6 if x > 6 else x)
    df.answer_links = df.answer_links.apply(lambda x: 10 if x > 10 else x)
    
    #TF-IDF
    text_features = 15
    for col in ['question_text', 'answer_text']:
        tfidf = TfidfVectorizer(max_features=text_features, norm='l2')
        tfidf.fit(df[col])
        tfidf_data = np.array(tfidf.transform(df[col]).toarray(), dtype=np.float16)

        for i in range(text_features):
            df[col + '_tfidf_' + str(i)] = tfidf_data[:, i]
        
        del tfidf, tfidf_data
        gc.collect()
        
    df.drop(['question_text', 'answer_text'], axis=1, inplace=True) #not using whole text
    
    #time based features
    df['time_till_answered'] = df.apply(lambda row: row['answer_utc'] - row['question_utc'], axis=1) #time from question till answer
    df['answer_dow'] = pd.to_datetime(df['answer_utc']).dt.dayofweek
    #df["time_trend"] = pd.to_datetime(df["answer_utc"] - df["answer_utc"].min()).dt.total_seconds()/3600
    
    #question counting
    question_count = df['question_id'].value_counts().to_dict()
    df['answer_count'] = df.apply(lambda row: question_count[row['question_id']], axis=1)
    
    #change subreddit into a numerical feature
    subreddit = pd.get_dummies(df['subreddit'])
    df.drop(['subreddit'], axis=1, inplace=True)
    df = df.join(subreddit)
    
    df.drop(['question_id'], axis=1, inplace=True) #remove question_id
    return df
from sklearn.preprocessing import StandardScaler

def scale(df):
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler.transform(df)
X_raw, y = load_data()
X = scale(feature_engineering(X_raw))

X_train, X_test, y_train, y_test = train_test_split(X, y)
gc.collect()

print(X_train.head())
print(X_test.head())
print('Making model')
model = sgdreg = SGDRegressor(penalty='elasticnet', alpha=0.15, n_iter=300, random_state=seed)
model.fit(X_train, np.log1p(y_train))
#model.fit(X_train, y_train)
gc.collect()
print('Trained model')
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(
        np.mean((np.log1p(y) - np.log1p(y0)) ** 2)
    )
y_train_theor = np.expm1(model.predict(X_train))
y_test_theor = np.expm1(model.predict(X_test))
#y_train_theor = model.predict(X_train)
#y_test_theor = model.predict(X_test)
print()
print("Training set")
print("RMSLE:   ", rmsle(y_train, y_train_theor))

print("Test set")
print("RMSLE:   ", rmsle(y_test, y_test_theor))
X_val_raw, _ = load_data('../input/kaggledays-warsaw/test.csv')
solution = pd.DataFrame(index=X_val_raw.index)
X_val = scale(feature_engineering(X_val_raw))

solution['answer_score'] = np.expm1(model.predict(X_val))
#solution['answer_score'] = model.predict(X_val)
solution['answer_score'] = solution.apply(lambda row: 0 if row['answer_score'] < 0 else row['answer_score'], axis=1) #hack for bad values :)
print('Saving data')
solution.to_csv('answer-' + str(time.time()) + '.csv', float_format='%.8f')
print('Saved data')
gc.collect()