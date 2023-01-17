import numpy as np
import random

seed = 2018
PYTHONHASHSEED = seed
random.seed(seed)
np.random.seed(seed)
import os
os.environ['OMP_NUM_THREADS'] = '4'
import pandas as pd
import lightgbm as lgb
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
X_raw, y = load_data()
X = feature_engineering(X_raw)
X_train, X_test, y_train, y_test = train_test_split(X, y)
gc.collect()

print(X_train.head())
print(X_test.head())
print('Making LGB datasets')
dtrain = lgb.Dataset(X_train.values, label=y_train.values)
dval = lgb.Dataset(X_test.values, label=y_test.values)
gc.collect()
print('LGB datasets ready')
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean((np.log1p(y) - np.log1p(y0)) ** 2))

def log_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', rmsle(labels, preds.clip(min=0)), False
params = {
    'boosting_type': 'gbdt',  # I think dart would be better, but takes too long to run
    # 'drop_rate': 0.09,  # Rate at which to drop trees
    'objective': 'regression',
    'metric': 'l2_root', #'rmsle',
    'learning_rate': 0.1,
    'num_leaves': 11,  # Was 255: Reduced to control overfitting
    'max_depth': 8,  # Was 8: LightGBM splits leaf-wise, so control depth via num_leaves
    'min_child_samples': 1,
    'min_bin': 1,
    'max_bin': 100,
    'subsample': 0.9,  # Was 0.7
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0.001,
    'subsample_for_bin': 200000,
    'min_split_gain': 0.00001,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01,
    'nthread': 4,
    'verbose': 0,
    'seed': seed,
    'early_stopping_round': 500
}
from sklearn.model_selection import RandomizedSearchCV

hypertune = False

if hypertune:
    print('Performing hypertune')
    
    # Create parameters to search
    gridParams = {
        'learning_rate': [0.005],
        'n_estimators': [8,16,24],
        'num_leaves': [6,8,12,16],
        'boosting_type' : ['gbdt'],
        'objective' : ['regression'],
        'random_state' : [seed],
        'colsample_bytree' : [0.64, 0.65, 0.66],
        'subsample' : [0.7,0.75],
        'reg_alpha' : [1,1.2],
        'reg_lambda' : [1,1.2,1.4],
    }
    
    mdl = lgb.LGBMClassifier(boosting_type= 'gbdt', 
          objective = 'regression', 
          n_jobs = 5, # Updated from 'nthread' 
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'], 
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'], 
          subsample_freq = params['subsample_freq'], 
          min_split_gain = params['min_split_gain'], 
          min_child_weight = params['min_child_weight'], 
          min_child_samples = params['min_child_samples'])

    # Create the grid
    grid = RandomizedSearchCV(mdl, gridParams, verbose=1, cv=5, n_jobs=-1)
    # Run the grid
    grid.fit(X_train, y_train)

    # Using parameters already set above, replace in the best from the grid search
    params['colsample_bytree'] = grid.best_params_['colsample_bytree']
    params['learning_rate'] = grid.best_params_['learning_rate'] 
    params['num_leaves'] = grid.best_params_['num_leaves']
    params['reg_alpha'] = grid.best_params_['reg_alpha']
    params['reg_lambda'] = grid.best_params_['reg_lambda']
    params['subsample'] = grid.best_params_['subsample']
print('Cross validation')
cv_result_lgb = lgb.cv(params, 
                       dtrain, 
                       num_boost_round=2000, 
                       nfold=5, 
                       stratified=False, 
                       verbose_eval=0,
                       show_stdv=True)
num_boost_rounds_lgb = len(cv_result_lgb['rmse-mean'])
print('num_boost_rounds_lgb=' + str(num_boost_rounds_lgb))
#print(cv_result_lgb)
print('Making model')
model = lgb.train(params=params, train_set=dtrain, valid_sets=dval, num_boost_round=num_boost_rounds_lgb, feval=log_error)
del dtrain, dval
gc.collect()
print('Trained model')

feat_imp = list(model.feature_importance())
cols = X_train.columns.values.tolist()

for feat_info in zip(cols, feat_imp):
    print('Column {} has {} importance'.format(feat_info[0], feat_info[1]))
#y_train_theor = np.expm1(model.predict(X_train))
#y_test_theor = np.expm1(model.predict(X_test))
y_train_theor = model.predict(X_train)
y_test_theor = model.predict(X_test)
print()
print("Training set")
print("RMSLE:   ", rmsle(y_train, y_train_theor))

print("Test set")
print("RMSLE:   ", rmsle(y_test, y_test_theor))
X_val_raw, _ = load_data('../input/kaggledays-warsaw/test.csv')
solution = pd.DataFrame(index=X_val_raw.index)
X_val = feature_engineering(X_val_raw)

#solution['answer_score'] = np.expm1(model.predict(X_val))
solution['answer_score'] = model.predict(X_val)
solution['answer_score'] = solution.apply(lambda row: 0 if row['answer_score'] < 0 else row['answer_score'], axis=1) #hack for bad values :)
print('Saving data')
solution.to_csv('answer-' + str(time.time()) + '.csv', float_format='%.8f')
print('Saved data')
gc.collect()