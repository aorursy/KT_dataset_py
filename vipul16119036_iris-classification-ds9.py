# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/iris/Iris.csv')
train.head()
train.info()
X = train.iloc[:, :-1]

y = train.iloc[:, -1]
plt.xlabel('Features')

plt.ylabel('Species')



pltX = train.loc[:, 'SepalLengthCm']

pltY = train.loc[:, 'Species']

plt.scatter(pltX,pltY, color='blue', label = 'SepalLengthCm')



pltX = train.loc[:, 'SepalWidthCm']

pltY = train.loc[:, 'Species']

plt.scatter(pltX,pltY, color='red', label = 'SepalWidthCm')



pltX = train.loc[:, 'PetalLengthCm']

pltY = train.loc[:, 'Species']

plt.scatter(pltX,pltY, color='green', label = 'PetalLengthCm')



pltX = train.loc[:, 'PetalWidthCm']

pltY = train.loc[:, 'Species']

plt.scatter(pltX,pltY, color='black', label = 'PetalWidthCm')



plt.legend(loc=4, prop={'size':7})

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print(y_pred)
y_test
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))