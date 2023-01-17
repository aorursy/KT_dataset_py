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
print(os.listdir('../input/nse-listed-1384-companies-data/'))
df = pd.read_csv('../input/nse-listed-1384-companies-data/MARUTI_data.csv')
df.head(5)
import seaborn as sns

sns.heatmap(df.corr())
df.isnull().sum()
X = df.iloc[:,[1,2,3]]

y = df.iloc[:,[4]]

X.fillna(X.mean(), inplace=True)

y.fillna(y.mean(), inplace=True)
sns.pairplot(x_vars=['open','high','low'], y_vars='close', data=df, height=7, aspect=0.7)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit(X_train,y_train)

print(f"score = {(reg.score(X_test, y_test))*100}")
from sklearn.metrics import median_absolute_error, mean_absolute_error, r2_score

y_pred = reg.predict(X_test)

print(f'{median_absolute_error(y_test,y_pred)},{mean_absolute_error(y_test,y_pred)}, {r2_score(y_test,y_pred)}')
reg.fit(X, y)

y_know = reg.predict([[7033,7014,6895],[6977,7156,6940],[7081,7450,7062]])

y_true =[[6975],[7124],[7302]]

r2_score(y_true,y_know)
reg.predict([[7333,7392,7277]])
def any_stock(stock_name, today_value=None):

    '''function to predict any stock values

    stock_name == str; today_value= list,[open,high,low]

    '''

    df = pd.read_csv('../input/nse-listed-1384-companies-data/historical_data/HISTORICAL_DATA/'+ stock_name)

    df.fillna(df.mean(),inplace=True)

    X = df.iloc[:,[1,2,3]]

    y = df.iloc[:,[4]]

    sns.pairplot(x_vars=['open','high','low'], y_vars='close', data=df, height=7, aspect=0.7)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

    reg = linear_model.LinearRegression()

    reg.fit(X_train,y_train)

    print(f"score = {(reg.score(X_test, y_test))*100}")

    y_pred = reg.predict(X_test)

    print(f'median_absolute_error ====== {median_absolute_error(y_test,y_pred)}')

    print(f'mean absoulte error ======= {mean_absolute_error(y_test,y_pred)}')

    print(f'r2 score ========== {r2_score(y_test,y_pred)}')

    if today_value != None:

        y_today = reg.predict([today_value])

        print(f"predicted closing price is = {y_today}")
today = [0.20,0.20,0.15]

any_stock('KGL_data.csv', today)