import os

import numpy as np

import pandas as pd

import time

from tqdm import tqdm



from sklearn.metrics import f1_score

from sklearn.model_selection import KFold

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from scipy.sparse import hstack

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

from tqdm import tqdm_notebook, tqdm

from scipy import stats



import nltk

from nltk.corpus import stopwords

import string

import gc



from scipy.sparse import hstack



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
import os

print(os.listdir("../input/google-quest-challenge"))
train = pd.read_csv('../input/google-quest-challenge/train.csv').fillna(' ')

test = pd.read_csv('../input/google-quest-challenge/test.csv').fillna(' ')

sample_submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
def spearman_corr(y_true, y_pred):

        if np.ndim(y_pred) == 2:

            corr = np.mean([stats.spearmanr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])])

        else:

            corr = stats.spearmanr(y_true, y_pred)[0]

        return corr
train.head()
test.head()
train.shape
test.shape
targets = list(sample_submission.columns[1:])

targets
train[targets].describe()
np.unique(train[targets].values, return_counts=True)
np.unique(train[targets].values).shape
x= np.unique(train['question_asker_intent_understanding'].values, return_counts=True)[0]

y= np.unique(train['question_asker_intent_understanding'].values, return_counts=True)[1]

plt.bar(x, y, align='center', width=0.05)
x= np.unique(train['question_body_critical'].values, return_counts=True)[0]

y= np.unique(train['question_body_critical'].values, return_counts=True)[1]

plt.bar(x, y, align='center', width=0.05)
x= np.unique(train['question_not_really_a_question'].values, return_counts=True)[0]

y= np.unique(train['question_not_really_a_question'].values, return_counts=True)[1]

plt.bar(x, y, align='center', width=0.05)
x= np.unique(train['question_conversational'].values, return_counts=True)[0]

y= np.unique(train['question_conversational'].values, return_counts=True)[1]

plt.bar(x, y, align='center', width=0.05)
corr = train[targets].corr()

corr.style.background_gradient(cmap='coolwarm')
eng_stopwords = set(stopwords.words("english"))





## Number of words in the text ##

train["question_title_num_words"] = train["question_title"].apply(lambda x: len(str(x).split()))

test["question_title_num_words"] = test["question_title"].apply(lambda x: len(str(x).split()))

train["question_body_num_words"] = train["question_body"].apply(lambda x: len(str(x).split()))

test["question_body_num_words"] = test["question_body"].apply(lambda x: len(str(x).split()))

train["answer_num_words"] = train["answer"].apply(lambda x: len(str(x).split()))

test["answer_num_words"] = test["answer"].apply(lambda x: len(str(x).split()))





## Number of unique words in the text ##

train["question_title_num_unique_words"] = train["question_title"].apply(lambda x: len(set(str(x).split())))

test["question_title_num_unique_words"] = test["question_title"].apply(lambda x: len(set(str(x).split())))

train["question_body_num_unique_words"] = train["question_body"].apply(lambda x: len(set(str(x).split())))

test["question_body_num_unique_words"] = test["question_body"].apply(lambda x: len(set(str(x).split())))

train["answer_num_unique_words"] = train["answer"].apply(lambda x: len(set(str(x).split())))

test["answer_num_unique_words"] = test["answer"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

train["question_title_num_chars"] = train["question_title"].apply(lambda x: len(str(x)))

test["question_title_num_chars"] = test["question_title"].apply(lambda x: len(str(x)))

train["question_body_num_chars"] = train["question_body"].apply(lambda x: len(str(x)))

test["question_body_num_chars"] = test["question_body"].apply(lambda x: len(str(x)))

train["answer_num_chars"] = train["answer"].apply(lambda x: len(str(x)))

test["answer_num_chars"] = test["answer"].apply(lambda x: len(str(x)))



## Number of stopwords in the text ##

train["question_title_num_stopwords"] = train["question_title"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

test["question_title_num_stopwords"] = test["question_title"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

train["question_body_num_stopwords"] = train["question_body"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

test["question_body_num_stopwords"] = test["question_body"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

train["answer_num_stopwords"] = train["answer"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

test["answer_num_stopwords"] = test["answer"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))



## Number of punctuations in the text ##

train["question_title_num_punctuations"] =train['question_title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test["question_title_num_punctuations"] =test['question_title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

train["question_body_num_punctuations"] =train['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test["question_body_num_punctuations"] =test['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

train["answer_num_punctuations"] =train['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test["answer_num_punctuations"] =test['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



## Number of title case words in the text ##

train["question_title_num_words_upper"] = train["question_title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test["question_title_num_words_upper"] = test["question_title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

train["question_body_num_words_upper"] = train["question_body"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test["question_body_num_words_upper"] = test["question_body"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

train["answer_num_words_upper"] = train["answer"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test["answer_num_words_upper"] = test["answer"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

features = ['question_title_num_words', 'question_body_num_words', 'answer_num_words', 'question_title_num_unique_words', 'question_body_num_unique_words', 'answer_num_unique_words',

           'question_title_num_chars', 'question_body_num_chars', 'answer_num_chars', 'question_title_num_stopwords', 'question_body_num_stopwords', 'question_title_num_punctuations',

           'question_body_num_punctuations', 'answer_num_punctuations', 'question_title_num_words_upper', 'question_body_num_words_upper', 'answer_num_words_upper']
plt.figure(figsize=(12,8))

sns.violinplot(data=train['question_body_num_words'])

plt.show()

plt.figure(figsize=(12,8))

sns.violinplot(data=train['question_body_num_chars'])

plt.show()


plt.figure(figsize=(12,8))

sns.violinplot(data=train['answer_num_chars'])

plt.show()
X_train = train[features].values

X_test = test[features].values

class_names_2 = [class_name+'_2' for class_name in targets]

for class_name in targets:

    train[class_name+'_2'] = (train[class_name].values >= 0.5)*1
%%time



submission_1 = pd.DataFrame.from_dict({'qa_id': test['qa_id']})



scores = []

spearman_scores = []



for class_name in tqdm_notebook(targets):

    print(class_name)

    Y = train[class_name]

    

    n_splits = 3

    kf = KFold(n_splits=n_splits, random_state=47)



    train_oof = np.zeros((X_train.shape[0], ))

    test_preds = 0

    

    score = 0



    for jj, (train_index, val_index) in enumerate(kf.split(X_train)):

        #print("Fitting fold", jj+1)

        train_features = X_train[train_index]

        train_target = Y[train_index]



        val_features = X_train[val_index]

        val_target = Y[val_index]



        model = Ridge()

        model.fit(train_features, train_target)

        val_pred = model.predict(val_features)

        train_oof[val_index] = val_pred

        #print("Fold auc:", roc_auc_score(val_target, val_pred))

        #score += roc_auc_score(val_target, val_pred)/n_splits



        test_preds += model.predict(X_test)/n_splits

        del train_features, train_target, val_features, val_target

        gc.collect()

        

    model = Ridge()

    model.fit(X_train, Y)

    

    preds = model.predict(X_test)

    mms = MinMaxScaler(copy=True, feature_range=(0, 1))

    preds = mms.fit_transform(preds.reshape(-1, 1)).flatten()

    submission_1[class_name] = (preds+0.00005)/1.0001

        

    score = roc_auc_score(train[class_name+'_2'], train_oof) 

    

    

    spearman_score = spearman_corr(train[class_name], train_oof)

    print("spearman_corr:", spearman_score)

    print("auc:", score, "\n")

    spearman_scores.append(spearman_score)

    

    scores.append(score)

    

print("Mean auc:", np.mean(scores))

print("Mean spearman_scores", np.mean(spearman_scores))
HistGradientBoostingRegressor()
%%time



submission_2 = pd.DataFrame.from_dict({'qa_id': test['qa_id']})



scores = []

spearman_scores = []



for class_name in tqdm_notebook(targets):

    print(class_name)

    Y = train[class_name]

    

    n_splits = 3

    kf = KFold(n_splits=n_splits, random_state=47)



    train_oof = np.zeros((X_train.shape[0], ))

    test_preds = 0

    

    score = 0



    for jj, (train_index, val_index) in enumerate(kf.split(X_train)):

        #print("Fitting fold", jj+1)

        train_features = X_train[train_index]

        train_target = Y[train_index]



        val_features = X_train[val_index]

        val_target = Y[val_index]



        model = HistGradientBoostingRegressor(max_depth=5)

        model.fit(train_features, train_target)

        val_pred = model.predict(val_features)

        train_oof[val_index] = val_pred

        #print("Fold auc:", roc_auc_score(val_target, val_pred))

        #score += roc_auc_score(val_target, val_pred)/n_splits



        test_preds += model.predict(X_test)/n_splits

        del train_features, train_target, val_features, val_target

        gc.collect()

        

    model = HistGradientBoostingRegressor(max_depth=5)

    model.fit(X_train, Y)

    

    preds = model.predict(X_test)

    mms = MinMaxScaler(copy=True, feature_range=(0, 1))

    preds = mms.fit_transform(preds.reshape(-1, 1)).flatten()

    submission_2[class_name] = (preds+0.00005)/1.0001

        

    score = roc_auc_score(train[class_name+'_2'], train_oof) 

    

    

    spearman_score = spearman_corr(train[class_name], train_oof)

    print("spearman_corr:", spearman_score)

    print("auc:", score, "\n")

    spearman_scores.append(spearman_score)

    

    scores.append(score)

    

print("Mean auc:", np.mean(scores))

print("Mean spearman_scores", np.mean(spearman_scores))
submission_1.head()
submission_2.head()
submission = submission_1.copy()

submission[targets] = 0.1*submission_1[targets].values + 0.9*submission_2[targets].values

submission.head()
submission.to_csv('submission.csv', index=False)