from sklearn import *

import numpy as np

import pandas as pd

import glob



data = {k.split('/')[-1][:-4]:k for k in glob.glob('/kaggle/input/**/**.csv')}

train = pd.read_csv(data['jigsaw-toxic-comment-train'], usecols=['id', 'comment_text', 'toxic'])

val = pd.read_csv(data['validation'], usecols=['comment_text', 'toxic'])

test = pd.read_csv(data['test'], usecols=['id', 'content'])

test.columns = ['id', 'comment_text']

test['toxic'] = 0.5



#https://www.kaggle.com/hamditarek/ensemble

sub2 = pd.read_csv('../input/submit/submission14.csv')
%%time

def f_experience(c, s):

    it = {'memory':10,

        'influence':0.5,

        'inference':0.5,

        'interest':0.9,

        'sentiment':1e-10,

        'harmony':0.5}

    

    exp = {}

    

    for i in range(len(c)):

        words = set([w for w in str(c[i]).lower().split(' ')])

        for w in words:

            try:

                exp[w]['influence'] = exp[w]['influence'][1:] + [s[i]] #need to normalize

                exp[w]['inference'] += 1

                exp[w]['interest'] = exp[w]['interest'][1:] + [(exp[w]['interest'][it['memory']-1] + (s[i] * it['interest']))/2]

                exp[w]['sentiment'] += s[i]

                #exp[w]['harmony']

            except:

                m = [0. for m_ in range(it['memory'])]

                exp[w] = {}

                exp[w]['influence'] = m[1:] + [s[i]]

                exp[w]['inference'] = 1

                exp[w]['interest'] = m[1:] + [s[i] * it['interest'] / 2]

                exp[w]['sentiment'] = s[i]

                #exp[w]['harmony'] = 0

                

    for w in exp:

        exp[w]['sentiment'] /= exp[w]['inference'] + it['sentiment']

        exp[w]['inference'] /= len(c) * it['inference']



    return exp



exp = f_experience(train['comment_text'].values, train['toxic'].values)
%%time

def features(df):

    df['len'] = df['comment_text'].map(len)

    df['wlen'] = df['comment_text'].map(lambda x: len(str(x).split(' ')))

    

    df['influence_sum'] = df['comment_text'].map(lambda x: np.sum([np.mean(exp[w]['influence']) if w in exp else 0 for w in str(x).lower().split(' ')]))

    df['influence_mean'] = df['comment_text'].map(lambda x: np.mean([np.mean(exp[w]['influence']) if w in exp else 0 for w in str(x).lower().split(' ')]))

    

    df['inference_sum'] = df['comment_text'].map(lambda x: np.sum([exp[w]['inference'] if w in exp else 0 for w in str(x).lower().split(' ')]))

    df['inference_mean'] = df['comment_text'].map(lambda x: np.mean([exp[w]['inference'] if w in exp else 0 for w in str(x).lower().split(' ')]))

    

    df['interest_sum'] = df['comment_text'].map(lambda x: np.sum([np.mean(exp[w]['interest']) if w in exp else 0 for w in str(x).lower().split(' ')]))

    df['interest_mean'] = df['comment_text'].map(lambda x: np.mean([np.mean(exp[w]['interest']) if w in exp else 0 for w in str(x).lower().split(' ')]))

    

    df['sentiment_sum'] = df['comment_text'].map(lambda x: np.sum([exp[w]['sentiment'] if w in exp else 0.5 for w in str(x).lower().split(' ')]))

    df['sentiment_mean'] = df['comment_text'].map(lambda x: np.mean([exp[w]['sentiment'] if w in exp else 0.5 for w in str(x).lower().split(' ')]))

    return df



val = features(val)

test= features(test)
col = [c for c in val if c not in ['id', 'comment_text', 'toxic']]

x1, x2, y1, y2 = model_selection.train_test_split(val[col], val['toxic'], test_size=0.3, random_state=20)



model = ensemble.ExtraTreesClassifier(n_estimators=1000, max_depth=7, n_jobs=-1, random_state=20)

model.fit(x1, y1)

print(metrics.roc_auc_score(y2, model.predict_proba(x2)[:,1].clip(0.,1.)))



model.fit(val[col], val['toxic'])

test['toxic'] = model.predict_proba(test[col])[:,1].clip(0.,1.)

sub1 = test[['id', 'toxic']]
sub1.rename(columns={'toxic':'toxic1'}, inplace=True)

sub2.rename(columns={'toxic':'toxic2'}, inplace=True)

sub3 = pd.merge(sub1, sub2, how='left', on='id')



sub3['toxic'] = (sub3['toxic1'] * 0.1) + (sub3['toxic2'] * 0.9) #blend 1

sub3['toxic'] = (sub3['toxic2'] * 0.5) + (sub3['toxic'] * 0.5) #blend 2



sub3[['id', 'toxic']].to_csv('submission.csv', index=False)
#Is it toxic :)

test = pd.DataFrame(['Howling with Wolf on Lügenpresse'], columns=['comment_text'])

test['id'] = test.index

test= features(test)

test['toxic'] = model.predict_proba(test[col])[:,1].clip(0.,1.)

test[['id', 'comment_text', 'toxic']].head()