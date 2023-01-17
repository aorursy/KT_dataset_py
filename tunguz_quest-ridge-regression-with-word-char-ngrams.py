# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from scipy.sparse import hstack

from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

from sklearn.preprocessing import MinMaxScaler



from tqdm import tqdm_notebook, tqdm

from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import gc

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def spearman_corr(y_true, y_pred):

        if np.ndim(y_pred) == 2:

            corr = np.mean([stats.spearmanr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])])

        else:

            corr = stats.spearmanr(y_true, y_pred)[0]

        return corr
train = pd.read_csv('../input/google-quest-challenge/train.csv').fillna(' ')

test = pd.read_csv('../input/google-quest-challenge/test.csv').fillna(' ')

train.head()
train.shape
test.shape
np.unique(train['category'].values)
train_text_1 = train['question_body']

test_text_1 = test['question_body']

all_text_1 = pd.concat([train_text_1, test_text_1])



train_text_2 = train['answer']

test_text_2 = test['answer']

all_text_2 = pd.concat([train_text_2, test_text_2])



train_text_3 = train['question_title']

test_text_3 = test['question_title']

all_text_3 = pd.concat([train_text_3, test_text_3])

sample_submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv').fillna(' ')

sample_submission.head()
class_names = list(sample_submission.columns[1:])

class_names
class_names_q = class_names[:21]

class_names_a = class_names[21:]

class_names_a
class_names_2 = [class_name+'_2' for class_name in class_names]

for class_name in class_names:

    train[class_name+'_2'] = (train[class_name].values >= 0.5)*1
%%time

word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 2),

    max_features=80000)

word_vectorizer.fit(all_text_1)

train_word_features_1 = word_vectorizer.transform(train_text_1)

test_word_features_1 = word_vectorizer.transform(test_text_1)



word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 2),

    max_features=80000)

word_vectorizer.fit(all_text_2)

train_word_features_2 = word_vectorizer.transform(train_text_2)

test_word_features_2 = word_vectorizer.transform(test_text_2)



word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 2),

    max_features=80000)

word_vectorizer.fit(all_text_3)

train_word_features_3 = word_vectorizer.transform(train_text_3)

test_word_features_3 = word_vectorizer.transform(test_text_3)



char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    ngram_range=(1, 4),

    max_features=47000)

char_vectorizer.fit(all_text_1)

train_char_features_1 = char_vectorizer.transform(train_text_1)

test_char_features_1 = char_vectorizer.transform(test_text_1)



char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    ngram_range=(1, 4),

    max_features=47000)

char_vectorizer.fit(all_text_2)

train_char_features_2 = char_vectorizer.transform(train_text_2)

test_char_features_2 = char_vectorizer.transform(test_text_2)



char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    ngram_range=(1, 4),

    max_features=47000)

char_vectorizer.fit(all_text_3)

train_char_features_3 = char_vectorizer.transform(train_text_3)

test_char_features_3 = char_vectorizer.transform(test_text_3)



train_features_1 = hstack([train_char_features_1, train_word_features_1, train_char_features_3, train_word_features_3])

test_features_1 = hstack([test_char_features_1, test_word_features_1, test_char_features_3, test_word_features_3])

train_features_2 = hstack([train_char_features_2, train_word_features_2])

test_features_2 = hstack([test_char_features_2, test_word_features_2])
train_features_1= train_features_1.tocsr()

train_features_2= train_features_2.tocsr()
alphas = {'question_asker_intent_understanding': 40,

         'question_body_critical':7,

          'question_conversational':35,

          'question_expect_short_answer':65,

          'question_fact_seeking':10,

          'question_has_commonly_accepted_answer':25,

          'question_interestingness_others':50,

          'question_interestingness_self':30,

          'question_multi_intent':7,

          'question_not_really_a_question':55,

          'question_opinion_seeking':15,

          'question_type_choice':4,

          'question_type_compare':30,

          'question_type_consequence':45,

          'question_type_definition':60,

          'question_type_entity':11,

          'question_type_instructions':6,

          'question_type_procedure':40,

          'question_type_reason_explanation':13,

          'question_type_spelling':1,

          'question_well_written':8,

          'answer_helpful':30,

          'answer_level_of_information':8,

          'answer_plausible':20,

          'answer_relevance':60,

          'answer_satisfaction':11,

          'answer_type_instructions':3,

          'answer_type_procedure':25,

          'answer_type_reason_explanation':3,

          'answer_well_written':25

         }
%%time

submission = pd.DataFrame.from_dict({'qa_id': test['qa_id']})



train_preds = []

test_preds = []

scores = []

spearman_scores = []



for class_name in tqdm_notebook(class_names_q):

    print(class_name)

    Y = train[class_name]

    

    n_splits = 3

    kf = KFold(n_splits=n_splits, random_state=47)



    train_oof_1 = np.zeros((train_features_1.shape[0], ))

    test_preds_1 = 0

    

    score = 0



    for jj, (train_index, val_index) in enumerate(kf.split(train_features_1)):

        #print("Fitting fold", jj+1)

        train_features = train_features_1[train_index]

        train_target = Y[train_index]



        val_features = train_features_1[val_index]

        val_target = Y[val_index]



        model = Ridge(alpha = alphas[class_name])

        model.fit(train_features, train_target)

        val_pred = model.predict(val_features)

        train_oof_1[val_index] = val_pred

        #print("Fold auc:", roc_auc_score(val_target, val_pred))

        #spearman_corr

        #score += roc_auc_score(val_target, val_pred)/n_splits



        test_preds_1 += model.predict(test_features_1)/n_splits

        del train_features, train_target, val_features, val_target

        gc.collect()

        

    model = Ridge(alpha = alphas[class_name])

    model.fit(train_features_1, Y)

    #print(test_preds_1.max())

    #print(test_preds_1.min())

    mms = MinMaxScaler(copy=True, feature_range=(0, 1))

    test_preds_1 = mms.fit_transform(test_preds_1.reshape(-1, 1)).flatten()

    preds = model.predict(test_features_1)

    mms = MinMaxScaler(copy=True, feature_range=(0, 1))

    preds = mms.fit_transform(preds.reshape(-1, 1)).flatten()

    submission[class_name] = (0.75*test_preds_1+0.25*preds+0.000005)/1.00001

    spearman_score = spearman_corr(train[class_name], train_oof_1)

    print("spearman_corr:", spearman_score) 

    spearman_scores.append(spearman_score)

    score = roc_auc_score(train[class_name+'_2'], train_oof_1)    

    print("auc:", score, "\n")

    train_preds.append(train_oof_1)

    test_preds.append(test_preds_1)

    scores.append(score)

    

    

%%time



for class_name in tqdm_notebook(class_names_a):

    print(class_name)

    Y = train[class_name]

    

    n_splits = 3

    kf = KFold(n_splits=n_splits, random_state=47)



    train_oof_2 = np.zeros((train_features_2.shape[0], ))

    test_preds_2 = 0

    

    score = 0



    for jj, (train_index, val_index) in enumerate(kf.split(train_features_1)):

        #print("Fitting fold", jj+1)

        train_features = train_features_2[train_index]

        train_target = Y[train_index]



        val_features = train_features_2[val_index]

        val_target = Y[val_index]



        model = Ridge(alpha = alphas[class_name])

        model.fit(train_features, train_target)

        val_pred = model.predict(val_features)

        train_oof_2[val_index] = val_pred

        #print("Fold auc:", roc_auc_score(val_target, val_pred))

        #score += roc_auc_score(val_target, val_pred)/n_splits



        test_preds_2 += model.predict(test_features_2)/n_splits

        del train_features, train_target, val_features, val_target

        gc.collect()

        

    model = Ridge(alpha = alphas[class_name])

    model.fit(train_features_2, Y)

    print(test_preds_2.max())

    

    preds = model.predict(test_features_2)

    mms = MinMaxScaler(copy=True, feature_range=(0, 1))

    test_preds_2 = mms.fit_transform(test_preds_2.reshape(-1, 1)).flatten()

    mms = MinMaxScaler(copy=True, feature_range=(0, 1))

    preds = mms.fit_transform(preds.reshape(-1, 1)).flatten()

    submission[class_name] = (0.75*test_preds_2+0.25*preds+0.000005)/1.00001

        

    score = roc_auc_score(train[class_name+'_2'], train_oof_2) 

    

    

    spearman_score = spearman_corr(train[class_name], train_oof_2)

    print("spearman_corr:", spearman_score)

    print("auc:", score, "\n")

    spearman_scores.append(spearman_score)

    

    train_preds.append(train_oof_2)

    test_preds.append(test_preds_2)

    scores.append(score)
print("Mean auc:", np.mean(scores))

print("Mean spearman_scores", np.mean(spearman_scores))
submission.to_csv('submission.csv', index=False)

submission.head()
submission[class_names].values.max()
submission[class_names].values.min()