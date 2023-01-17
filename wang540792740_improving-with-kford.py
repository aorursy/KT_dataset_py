import pandas as pd

import spacy

from sklearn.model_selection import KFold

import numpy as np

nlp = spacy.load('en_core_web_sm')

from sklearn import *

import warnings

warnings.filterwarnings('ignore')





test = pd.read_csv('../input/test_stage_2.tsv', delimiter='\t').rename(columns={'A': 'A_Noun', 'B': 'B_Noun'})

sub = pd.read_csv('../input/sample_submission_stage_2.csv')

test.shape, sub.shape




gh_test = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv", delimiter='\t')

gh_valid = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv", delimiter='\t')

train = pd.concat((gh_test, gh_valid)).rename(columns={'A': 'A_Noun', 'B': 'B_Noun'}).reset_index(drop=True)



def name_replace(s, r1, r2):

    s = str(s).replace(r1, r2)

    for r3 in r1.split(' '):

        s = str(s).replace(r3, r2)

    return s





def get_features(df):

    df['section_min'] = df[['Pronoun-offset', 'A-offset', 'B-offset']].min(axis=1)

    df['Pronoun-offset2'] = df['Pronoun-offset'] + df['Pronoun'].map(len)

    df['A-offset2'] = df['A-offset'] + df['A_Noun'].map(len)

    df['B-offset2'] = df['B-offset'] + df['B_Noun'].map(len)

    df['section_max'] = df[['Pronoun-offset2', 'A-offset2', 'B-offset2']].max(axis=1)

    df['Text'] = df.apply(lambda r: name_replace(r['Text'], r['A_Noun'], 'subjectone'), axis=1)

    df['Text'] = df.apply(lambda r: name_replace(r['Text'], r['B_Noun'], 'subjecttwo'), axis=1)



    df['A-dist'] = (df['Pronoun-offset'] - df['A-offset']).abs()

    df['B-dist'] = (df['Pronoun-offset'] - df['B-offset']).abs()

    return (df)



train = get_features(train)

test = get_features(test)



def get_nlp_features(s, w):

    doc = nlp(str(s))

    # print(doc)

    tokens = pd.DataFrame([[token.text, token.dep_] for token in doc], columns=['text', 'dep'])

    # print(tokens)

    # token.text is the word, token.dep is the characteristic of a word

    # print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])

    # print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

    # tokens == 'poss' means possessive.

    return len(tokens[((tokens['text']==w) & (tokens['dep']=='poss'))])

train['A-poss'] = train['Text'].map(lambda x: get_nlp_features(x, 'subjectone'))

# print(train['A-poss'])

# 2453    2

# Name: A-poss, Length: 2454, dtype: int64



train['B-poss'] = train['Text'].map(lambda x: get_nlp_features(x, 'subjecttwo'))

test['A-poss'] = test['Text'].map(lambda x: get_nlp_features(x, 'subjectone'))

test['B-poss'] = test['Text'].map(lambda x: get_nlp_features(x, 'subjecttwo'))



train = train.rename(columns={'A-coref':'A', 'B-coref':'B'})

train['A'] = train['A'].astype(int)

train['B'] = train['B'].astype(int)

train['NEITHER'] = 1.0 - (train['A'] + train['B'])



col = ['Pronoun-offset', 'A-offset', 'B-offset', 'section_min', 'Pronoun-offset2', 'A-offset2', 'B-offset2', 'section_max', 'A-poss', 'B-poss', 'A-dist', 'B-dist']

x1, x2, y1, y2 = model_selection.train_test_split(train[col].fillna(-1), train[['A', 'B', 'NEITHER']], test_size=0.2, random_state=1)

print(x1.head())



model = multiclass.OneVsRestClassifier(ensemble.RandomForestClassifier(max_depth = 7, n_estimators=1000, random_state=33))



folds = 5

xchange = 0.94

kf = KFold(n_splits=folds, shuffle=False, random_state=11)

trn = train[col].fillna(-1)

val = train[["A", "B", "NEITHER"]]

scores = []



for train_index, test_index in kf.split(train):

    x1, x2 = trn.iloc[train_index], trn.iloc[test_index]

    y1, y2 = val.iloc[train_index], val.iloc[test_index]



    model.fit(x1, y1)

    score = metrics.log_loss(y2, model.predict_proba(x2))

    if score < xchange:

        print("log-loss:", score)

    scores.append(score)



# print("CV Score(log-loss):", np.mean(scores))

model.fit(x1, y1)



# print('log_loss', metrics.log_loss(y2, model.predict_proba(x2)))

model.fit(train[col].fillna(-1), train[['A', 'B', 'NEITHER']])

results = model.predict_proba(test[col])

test['A'] = results[:,0]

test['B'] = results[:,1]

test['NEITHER'] = results[:,2]

test[['ID', 'A', 'B', 'NEITHER']].to_csv('submission.csv', index=False)





# 做可视化.

# Feature

# 调参Randomforest建议

# Xgbootst的参数

# https://www.cnblogs.com/zhizhan/p/5826089.html

# Grid Search 的过程来确定一组最佳的参数。其实这个过程说白了就是根据给定的参数候选对所有的组合进行暴力搜索。

# 用 20 个不同的随机种子来生成 Ensemble，最后取 Weighted Average。这个其实算是一种变相的 Bagging。

# 其意义在于按我实现 Stacking 的方式，我在训练 Base Model 时只用了 80% 的训练数据，而训练第二层的 Model 时用了 100% 的数据，

# 这在一定程度上增大了 Overfitting 的风险。而每次更改随机种子可以确保每次用的是不同的 80%，这样在多次训练取平均以后就相当于逼近了使用 100% 数据的效果