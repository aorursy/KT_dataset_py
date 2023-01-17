import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
data = pd.read_csv("../input/creditcardfraud/creditcard.csv")

data.head()
sns.distplot(data['Time'])
sns.countplot('Class', data=data)
data.sample(frac=1)



fraud_data = data.loc[data['Class'] == 1]

nonfraud_data = data.loc[data['Class'] == 0][0: 492]



normal_distributed = pd.concat([fraud_data, nonfraud_data])
normal_distributed.head()
sns.countplot('Class', data=normal_distributed)
# Dropping the Class in order it as our labels:

y = normal_distributed['Class']

normal_distributed.drop(['Class'], axis=1, inplace=True)
# Dividing the data into training and test sets:



train, test, ytrain, ytest = train_test_split(normal_distributed, y, train_size=0.7, test_size=0.3)



train = train.values

test = test.values

ytrain = ytrain.values

ytest = ytest.values
Model = SVC(random_state=0)
Model.fit(train, ytrain)
Preds = Model.predict(test)
print('The accuracy of the Model is:', accuracy_score(ytest, Preds))