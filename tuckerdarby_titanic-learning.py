# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))





def accuracy(truth, pred):

    return "Prediction accuracy: {:.2f}%".format((truth == pred).mean()*100)





def prepare(data, training):

    if training:

        subject = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']]

    else:

        subject = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]

    genders = pd.get_dummies(subject['Sex'])

    subject = subject.drop('Sex', axis=1)

    subject['male'] = genders['male']

    subject['female'] = genders['female']

    subject = subject.fillna(value=0)

    if training:

        outcomes = subject['Survived']

        subject = subject.drop('Survived', axis=1)

        return subject, outcomes

    return subject





test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')



#print(test['PassengerId'])





X, Y = prepare(train, True)

#Linear Regression Classifier

lr = LogisticRegression(penalty='l1', random_state=0)

lr.fit(X, Y)

#SVM Classifier

svm = SVC()

svm.fit(X, Y)



prepared_test = prepare(test, False)

preds_lr = lr.predict(prepared_test)

preds_svm = svm.predict(prepared_test)



results = pd.DataFrame(test['PassengerId'])

results['Survived'] = preds_svm



print(results)

# Any results you write to the current directory are saved as output.

results.to_csv('results.csv', index=False)