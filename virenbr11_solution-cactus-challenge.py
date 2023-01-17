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
# import required libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# plot stylings

plt.style.use('fivethirtyeight')

%matplotlib inline



# do not display warnings in notebook 

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/cactus-challenge/Python Test_conditions.csv")

df
test_df = pd.read_csv('../input/cactus-challenge/Python Test Files - Sudoku_attemps.csv')

test_df
# Glancing the columns 

df.info()
# Glancing the columns 

test_df.info()
df.isnull().sum()
test_df.isnull().sum()
sud=test_df.groupby('Game ID').sum()

sud
test_df = test_df.set_index('Batch ID')

test_df
train_df = pd.merge(test_df, df, on='Batch ID')

train_df = train_df.set_index('Batch ID')

train_df
train_df = train_df.replace({'Game ID':{'Sudoku #1': '1','Sudoku #2': '2','Sudoku #3': '3',

                                       'Sudoku #4': '4','Sudoku #5': '5','Sudoku #6': '6',

                                       'Sudoku #7': '7','Sudoku #8': '8','Sudoku #9': '9',

                                       'Sudoku #10': '10','Sudoku #11': '11','Sudoku #12': '12',

                                       'Sudoku #13': '13','Sudoku #14': '14','Sudoku #15': '15',

                                       'Sudoku #16': '16','Sudoku #17': '17','Sudoku #18': '18',

                                       'Sudoku #19': '19','Sudoku #20': '20','Sudoku #21': '21',

                                       'Sudoku #22': '22','Sudoku #23': '23','Sudoku #24': '24',

                                       'Sudoku #25': '25','Sudoku #26': '26'}})

train_df
x = train_df["Completed"].sum()

y = train_df["Tried"].sum()

labels = "Success", "Attempts"

explode = (0.1, 0)

fig1, ax1 = plt.subplots()

ax1.pie([x,y], explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  

plt.title("Number of attempts vs Success")

plt.show()
train_df.groupby(['Tried','Completed'])['Completed'].count()
f,ax=plt.subplots(2,1,figsize=(12,12))

train_df[['Tried','Completed']].groupby(['Tried']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Completed vs Tried')

sns.countplot('Tried',hue='Completed',data=train_df,ax=ax[1])

ax[1].set_title('Completed vs Tried')

plt.show()
cols = ['Soln D Concentration', 'Soln C Concentration', 'Soln B Concentration', 'Soln A Concentration', 'Relative Humidity', 'White Noise (db)']



sns.pairplot(df[cols], size=2.5)

plt.tight_layout()

plt.show()
cm = np.corrcoef(df[cols].values.T)

hm = sns.heatmap(cm,

                 cbar=True,

                 annot=True,

                 square=True,

                 fmt='.2f',

                 annot_kws={'size': 10},

                 yticklabels=cols,

                 xticklabels=cols)



plt.tight_layout()

plt.show()
sns.violinplot("Soln A Concentration", hue="Completed", data=train_df, split=True)

plt.title('Soln A Concentration vs Successful attempts')
sns.violinplot("Soln B Concentration", hue="Completed", data=train_df, split=True)

plt.title('Soln B Concentration vs Successful attempts')
sns.violinplot("Soln C Concentration", hue="Completed", data=train_df, split=True)

plt.title('Soln C Concentration vs Successful attempts')
sns.violinplot("Soln D Concentration", hue="Completed", data=train_df, split=True)

plt.title('Soln D Concentration vs Successful attempts')
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X = train_df.drop("Completed",axis=1)

Y = train_df['Completed']



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import metrics
regr = LinearRegression()



# Train the model using the training sets

regr.fit(X_train, y_train)



# Make predictions using the testing set

y_pred = regr.predict(X_test)



# The intercept

print("Intercept: ", regr.intercept_)

# The coefficients

print("Coefficients: ", regr.coef_[0])

# The mean absolute error

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

# The mean squared error

print("Mean Squared Error: ", metrics.mean_squared_error(y_test, y_pred))  

# The root mean squared error

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Explained variance score: 1 is perfect prediction

print('Variance score: ', r2_score(y_test, y_pred))
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df1