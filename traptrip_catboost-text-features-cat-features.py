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
import time 



from catboost import CatBoostClassifier, Pool

from transformers import BertTokenizerFast



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from tqdm.notebook import tqdm



pd.set_option('display.max_columns', 1000)

pd.set_option('display.max_rows', 1000)
tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased-conversational')
train = pd.read_csv('/kaggle/input/ml-guild-classification-task/train.csv')

test = pd.read_csv('/kaggle/input/ml-guild-classification-task/test.csv').drop('Unnamed: 0', axis=1)
train['title'].fillna('', inplace=True)

test['title'].fillna('', inplace=True)



# train['title'] = train['title'].astype(str)

# train['review'] = train['review'].astype(str)

# train['film_country'] = train['film_country'].astype(str)



# test['title'] = test['title'].astype(str)

# test['review'] = test['review'].astype(str)

# test['film_country'] = test['film_country'].astype(str)



# train['title'][train.title == 'nan'] = ''

# train['review'][train.review == 'nan'] = ''

# test['title'][test.title == 'nan'] = ''

# test['review'][test.review == 'nan'] = ''
texts = train['title'] + train['review']

pretrained_bpe_texts = [' '.join(map(str, tokenizer.encode(text))) for text in tqdm(texts)]
texts_test = test['title'] + test['review']

pretrained_bpe_texts_test = [' '.join(map(str, tokenizer.encode(text))) for text in tqdm(texts_test)]
labels = ['positive', 'negative', 'neutral']

X = train.drop(columns=['positive', 'negative', 'neutral'])

X_test = test.copy()
n_fold = 5

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=1)
def train_model_and_predict(X, X_test, y, folds, model_type='cat', 

                            categorical_cols = ['film_country'], 

                            text_cols=[]):

    prediction = np.zeros((len(X_test), 3))

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        print('Fold', fold_n, 'started at', time.ctime())

        X_train, X_valid = X.loc[train_index], X.loc[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

            

        if model_type == 'cat':

            train_pool = Pool(

                X_train, 

                y_train, 

                cat_features=categorical_cols,

                text_features=text_cols,

                feature_names=list(X_train)

            )

            valid_pool = Pool(

                X_valid, 

                y_valid, 

                cat_features=categorical_cols,

                text_features=text_cols,

                feature_names=list(X_valid)

            )

            model = CatBoostClassifier(

                                       task_type="GPU",

                                       devices='0:1'

            )

            model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=False)



            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test)

            

        scores.append(roc_auc_score(y_valid, y_pred_valid, multi_class='ovr'))

        print(f'Score: {scores[-1]:.4f}.')

        prediction += y_pred

        

        

    prediction /= n_fold

    print(f'CV mean score: {np.mean(scores):.4f}, std: {np.std(scores):.4f}.')

            

    return prediction
X['bpe_text'] = pretrained_bpe_texts

X_test['bpe_text'] = pretrained_bpe_texts_test
X_new = X.copy()

X_test_new = X_test.copy()
def get_day_part(x):

    hour = int(x.split(' ')[-1].split(':')[0])

    return 0 if hour > 5 and hour <= 11 else 1 if hour > 11 and hour <= 16 else 2 if hour > 16 else 3    

X_new['day'] = X_new['date'].apply(lambda x: int(x.split(' ')[0]))

X_new['month'] = X_new['date'].apply(lambda x: x.split(' ')[1])

X_new['day_part'] = X_new['date'].apply(get_day_part)



X_test_new['day'] = X_test_new['date'].apply(lambda x: int(x.split(' ')[0]))

X_test_new['month'] = X_test_new['date'].apply(lambda x: x.split(' ')[1])

X_test_new['day_part'] = X_test_new['date'].apply(get_day_part)
X_new['useful_avr'] = X_new['useful'] - X_new['useless']

X_test_new['useful_avr'] = X_test_new['useful'] - X_test_new['useless']



X_new['useful_share'] = X_new['useful'] / (X_new['useful'] + X_new['useless'])

X_test_new['useful_share'] = X_test_new['useful'] / (X_test_new['useful'] + X_test_new['useless'])
X_new = X_new.drop('date', axis=1)

X_test_new = X_test_new.drop('date', axis=1)
pred_cb = train_model_and_predict(X_new, X_test_new, 

                                  train[labels].to_numpy().argmax(axis=1), 

                                  folds=folds, model_type='cat', 

                                  categorical_cols=['film_year', 'film_country', 'month', 'day_part'], 

                                  text_cols=['review', 'title', 'bpe_text']

                                 )
ss = pd.read_csv('/kaggle/input/ml-guild-classification-task/sample_submission.csv')

ss[labels] = pred_cb
ss.to_csv('submission_catboost_bpe.csv', index=False)