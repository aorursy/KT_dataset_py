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
# Import necessary lybraries

import numpy as np

from scipy.stats import t

from scipy.stats import norm

import matplotlib.pyplot as plt

from scipy import stats

import pandas as pd

import io

import requests

import seaborn as sns



%matplotlib inline
data=pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')

data.columns=('age', 'gender', 'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphotase', 'alamine_aminotransferase', 'aspartate_aminotransferase', 'total_protiens', 'albumin', 'albumin_and_globulin_ratio', 'dataset')

data.tail()
data.shape
data.isnull().sum()
data['albumin_and_globulin_ratio'].mean()
data = data.fillna(0.94)
data.isnull().sum()
data['gender'] = data['gender'].apply(lambda x:1 if x=='Male' else 0)

data.head()
data['gender']
data.sort_values(by=['direct_bilirubin'], inplace=True)



X = data[['age', 'gender', 'direct_bilirubin', 'alkaline_phosphotase', 'alamine_aminotransferase', 'aspartate_aminotransferase', 'total_protiens', 'albumin', 'albumin_and_globulin_ratio', 'dataset']]

tb = data[['total_bilirubin']]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X,

    tb,

    test_size = 0.2,

    random_state = 42

)
X_train_db = X_train[['direct_bilirubin']]

X_test_db = X_test[['direct_bilirubin']]
X_train_db = np.array(X_train_db).reshape(-1, 1)

X_test_db = np.array(X_test_db).reshape(-1, 1)

y_train = np.array(y_train).reshape(-1, 1)

y_test = np.array(y_test).reshape(-1, 1)
from sklearn.linear_model import LinearRegression

model = LinearRegression()





model.fit(X_train_db, y_train)
model.score(X_test_db, y_test)
plt.xlabel('total_bilirubin')

plt.ylabel('direct_bilirubin')

plt.title('linear regression total_bilirubin vs direct_bilirubin')

x = np.linspace(min(X_train_db), max(X_train_db), 100)

y = (model.coef_*x + model.intercept_)

plt.plot(x,y)

plt.plot(X_train_db, y_train, 'r.')

plt.plot(X_test_db, y_test, 'g.')

plt.legend(['linear regression', 'training data', 'testing data'])

plt.show()
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)

X_poly = poly_reg.fit_transform(X_train_db)

pol_reg = LinearRegression()

pol_reg.fit(X_poly, y_train)

pol_reg.score(poly_reg.fit_transform(X_test_db), y_test)
def viz_polymonial():

    plt.scatter(X_train_db, y_train, c='red')

    plt.scatter(X_test_db, y_test, c='green')

    a = np.linspace(min(X_test_db), max(X_train_db), 100)

    b = pol_reg.predict(poly_reg.fit_transform(a))

    plt.plot(a, b, 'b')

    plt.title('Polynom regression')

    plt.xlabel('total_bilirubin')

    plt.ylabel('direct_bilirubin')

    plt.legend(['polynom regression', 'training data', 'testing data'])

    plt.show()

    return

viz_polymonial()
corr = data.corr()

corr = corr.round(2)

sns.heatmap(corr.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(10,10)

plt.show()
x = data.drop('dataset', axis=1)

y = data.dataset
from sklearn.linear_model import LogisticRegression



x_train, x_test, y_train, y_test = train_test_split(

    x, 

    y, 

    test_size=0.27, 

    random_state=27)
model = LogisticRegression(solver='liblinear').fit(x_train, y_train)
from sklearn.metrics import classification_report



model_pred = model.predict(x_test)



print(classification_report(y_test, model_pred))
people_before_50 = data[data['age'] < 50]

people_after_50 = data[data['age'] >= 50]



plt.hist(people_before_50['dataset'])

plt.hist(people_after_50['dataset'])

plt.title('age difference between iil and health people')

plt.xlabel('ill or not')

plt.ylabel('count')

plt.legend(['before_50','after_50'])

plt.show()