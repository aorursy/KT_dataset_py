import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

p = '../input/cat-in-the-dat-ii/'

X = pd.concat([pd.read_csv(p+'train.csv').iloc[:,1:-1],

               pd.read_csv(p+'test.csv').iloc[:,1:]]).astype('str')

y = pd.read_csv(p+'train.csv').target

sample = pd.read_csv(p+'sample_submission.csv') 

X = OneHotEncoder().fit_transform(X)

train,test = X[:600000],X[600000:]

sample['target'] = LogisticRegression(max_iter=1e5,C=0.06).fit(train,y).predict_proba(test)[:,1]

sample.to_csv('submission.csv',index=False)