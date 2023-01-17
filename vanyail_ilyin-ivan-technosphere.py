import numpy as np 

import pandas as pd

import time

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer
df_train = pd.read_csv('../input/train.csv')

df_train = df_train.dropna(how="any").reset_index(drop=True)
Word_Extractor = CountVectorizer(analyzer='char', ngram_range=(1,2), binary=True, lowercase=True)

Word_Extractor.fit(pd.concat((df_train.ix[:,'question1'],df_train.ix[:,'question2'])).unique())
question_1 = Word_Extractor.transform(df_train.ix[:,'question1'])

question_2 = Word_Extractor.transform(df_train.ix[:,'question2'])

y = np.array(df_train.ix[:,'is_duplicate'])
X = -(question_1 != question_2).astype(int)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
a = 0.165 / 0.37

b = (1 - 0.165) / (1 - 0.37)

# a - отношение единиц в test к единицам в train.
#попробуем случайный лес

t = time.clock()

parameters = {'n_estimators':range(10, 111, 20), 'max_depth':range(10,21,5)}

rf = RandomForestClassifier()

model = GridSearchCV(rf,parameters)

model.fit(X[:20000,:],y[:20000])

t = time.clock()-t

print(t)
print(model.best_params_)

print(model.best_score_)
t = time.clock()

model = RandomForestClassifier(n_estimators = 70, max_depth = 20, class_weight={1: a, 0: b})

model.fit(X,y)

t = time.clock()-t

print(t)

#При больших n и max_depth считается очень долго.

#Поэтому в поставим поменьше
t = time.clock()

df_test = pd.read_csv('../input/test.csv')

df_test.ix[df_test['question1'].isnull(),['question1','question2']] = 'random empty question'

df_test.ix[df_test['question2'].isnull(),['question1','question2']] = 'random empty question'

test_question_1 = Word_Extractor.transform(df_test.ix[:,'question1'])

test_question_2 = Word_Extractor.transform(df_test.ix[:,'question2'])

X_test = -(test_question_1 != test_question_2).astype(int)

t = time.clock()-t

print(t)
testPredictions = model.predict_proba(X_test)[:,1]