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

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

train_data.corr(method='pearson')

train_data = train_data.replace(to_replace = ['male','female'],value = [1,0])

train_data = train_data.fillna(0)
x = train_data.drop(['Survived','Name','Ticket','Cabin','Embarked'],axis = 1)



print(x)

y = train_data['Survived']

print(y.head())



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix,accuracy_score



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state = 1)

logmodel = LogisticRegression()

logmodel.fit(x_train,y_train)

predictions = logmodel.predict(x_test)

print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))

print(accuracy_score(y_test, predictions))
import seaborn as sns

import matplotlib.pyplot as plt

sns.heatmap(pd.DataFrame(confusion_matrix(y_test,predictions)))

plt.show()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")





test_data = test_data.replace(to_replace = ['male','female'],value = [1,0])

test_data = test_data.fillna(0)

test_data.head()
x = test_data.drop(['Name','Ticket','Cabin','Embarked'],axis = 1)



print(x)

y =  logmodel.predict(x)


