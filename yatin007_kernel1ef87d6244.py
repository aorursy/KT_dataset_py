# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("../input/predict-the-income-bi-hack/train.csv")
df.head(10)
for k, v in df.nunique().to_dict().items():

    print('{}={}'.format(k,v))
plt.scatter(df.Gender, df.Income, color = 'blue')
df.hist()
plt.scatter(df['Income'], df['Age'])
df.head()
df = df.drop(['ID', 'Race', 'Marital_Status', 'Relationship'], axis = 1)
df.head()
dummies = pd.get_dummies(df)

dummies
y = dummies['Income_>50K']
X = dummies.drop(['Income_>50K', 'Income_<=50K'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from time import time

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):

  results = {}

  start = time()

  learner= learner.fit(X_train[:sample_size],y_train[:sample_size])

  end = time()

  

  results['train_time'] = end- start

  

  start = time()

  pred_test=learner.predict(X_test)

  pred_train=learner.predict(X_train)

  end= time()

  

  results['pred_time'] = end-start

  

  results['acc_train'] = accuracy_score(y_train, pred_train)

  results['acc_test'] = accuracy_score(y_test, pred_test)

  

  results['f_train'] = fbeta_score(y_train, pred_train, beta = 0.5)

  results['f_test'] = fbeta_score(y_test, pred_test, beta = 0.5)

  

  return(results)


from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier



model1=LogisticRegression(random_state=0)

model2=GradientBoostingClassifier(random_state=0)

model3=RandomForestClassifier(random_state=0)
from sklearn.metrics import fbeta_score
results={}

sample_1=int(len(y_train)*0.05)

sample_10=int(len(y_train)*0.1)

sample_100=len(y_train)



for model in [model1, model2, model3]:

  # Getting model name

  model_name=model.__class__.__name__

  results[model_name]={}

  for i, samples in enumerate([sample_1, sample_10, sample_100]):

    results[model_name][i]=train_predict(model, samples, X_train, y_train, X_test, y_test)
print(results)
model3_imp = model3.fit(X_train, y_train).feature_importances_
y_train.value_counts()
4580/(14620+4580)
sub_df = pd.read_csv("../input/predict-the-income-bi-hack/sampleSubmission.csv")
sub_df.to_csv('submission.csv',index=False)