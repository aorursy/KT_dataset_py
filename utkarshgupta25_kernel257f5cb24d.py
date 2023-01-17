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
df = pd.read_csv("/kaggle/input/abalone.data", header = None)



df.head()
df.columns = ['Gender', 'Length', 'Diameter', 'Height', 'Whole_Weight', 'Shucked_Weight', 'Vescara_Weight', 'Shell_Weight', 'Rings']



df.head()
df.isna().sum()
df.describe()
df.corr()
import seaborn as sns

import matplotlib.pyplot as plt



sns.heatmap(df.corr(), cmap = "Purples", annot = True)
df['Rings'].unique()
for i in df.drop(columns = ['Rings']):

    

    if i != 'Gender':

        sns.regplot(df[i], df['Rings'])

    else:

        sns.boxplot(df['Gender'], df['Rings'])

        

    plt.show()
df['isInfant'] = df['Gender'].apply(lambda x: 1 if x == 'I' else 0)



df[['Rings', 'isInfant']].corr()
sns.boxplot(df['isInfant'], df['Rings'])
df = pd.get_dummies(df)



df.head()
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression as lr



x = df.drop(columns = ['Rings', 'isInfant'])

y = df['Rings']



kf = KFold(n_splits = 5)



train_id = []

test_id = []



x_arr = np.array(x)

y_arr = np.array(y)



for train_index, test_index in kf.split(x_arr):

    

    train_id.append(train_index)

    test_id.append(test_index)

from sklearn.ensemble import GradientBoostingRegressor as gbr



training_features = []

testing_features = []



training_label = []

testing_label = []



scores = []



for i in range(5):



    for index in train_id[i]:

        training_features.append(x_arr[index])

        testing_features.append(y_arr[index])



    for index in test_id[i]:



        training_label.append(x_arr[index])

        testing_label.append(y_arr[index])

        

    x_train= pd.DataFrame(training_features)

    y_train = pd.DataFrame(testing_features)

    

    x_test = pd.DataFrame(training_label)

    y_test = pd.DataFrame(testing_label)

    

    regressor = gbr(n_estimators = 500)



    regressor.fit(x_train, y_train)



    scores.append(regressor.score(x_test, y_test))

    

print(scores)

    

sum(scores)/5
from sklearn.model_selection import train_test_split as tts



scores = []



for i in range(500):



    x_train, x_test, y_train, y_test = tts(x,y, random_state = i)



    regressor.fit(x_train, y_train)



    scores.append(regressor.score(x_test, y_test))

    

sum(scores)/500
from sklearn.linear_model import Ridge



scores = []



for i in range(500):



    x_train, x_test, y_train, y_test = tts(x,y, random_state = i)

    

    regressor = Ridge()



    regressor.fit(x_train, y_train)



    scores.append(regressor.score(x_test, y_test))

    

sum(scores)/500
from sklearn.ensemble import RandomForestRegressor as rfr



scores = []



for i in range(100):



    x_train, x_test, y_train, y_test = tts(x,y, random_state = i)

    

    regressor = rfr(n_estimators = 10)



    regressor.fit(x_train, y_train)



    scores.append(regressor.score(x_test, y_test))

    

sum(scores)/100
from sklearn.ensemble import GradientBoostingRegressor as gbr



scores = []



for i in range(1):



    x_train, x_test, y_train, y_test = tts(x,y, random_state = i)

    

    regressor = gbr(n_estimators = 500, )



    regressor.fit(x_train, y_train)



    scores.append(regressor.score(x_test, y_test))

    

sum(scores)/1
from xgboost import XGBRegressor as xgbr



scores = []



for i in range(1):



    x_train, x_test, y_train, y_test = tts(x,y, random_state = i)

    

    regressor = xgbr(n_estimators = 10)



    regressor.fit(x_train, y_train)



    scores.append(regressor.score(x_test, y_test))

    

sum(scores)/1

    
x_train, x_test, y_train, y_test = tts(x,y, random_state = 0)

regressor = xgbr(n_estimators = 200)



regressor.fit(x_train, y_train)



regressor.score(x_test, y_test)

from sklearn.svm import SVR



scores = []



for i in range(1):



    x_train, x_test, y_train, y_test = tts(x,y, random_state = i)

    

    regressor = SVR(gamma = 'auto')



    regressor.fit(x_train, y_train)



    scores.append(regressor.score(x_test, y_test))

    

sum(scores)/1
print("HI")
x_train, x_test, y_train, y_test = tts(x,y, random_state = 20)

regressor = xgbr(n_estimators = 10)



regressor.fit(x_train, y_train)



regressor.score(x_test, y_test)