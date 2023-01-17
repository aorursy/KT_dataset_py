# Librairies

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

import random

%matplotlib inline
print(np.float64(0)/0)

print(np.Infinity-np.Infinity)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/boston-housing-dataset/HousingData.csv")
print("The shape of the dataset is {} rows and {} columns".format(df.shape[0],df.shape[1]))
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, train_size=0.7)
print("The shape of the train dataset is {} rows and {} columns (70% of the data)".format(train_df.shape[0],train_df.shape[1]))

print("The shape of the test dataset is {} rows and {} columns (30% of the data)".format(test_df.shape[0],test_df.shape[1]))
train_df.isnull().sum().sort_values(ascending=False)/len(train_df)*100
null_values_train = train_df.isnull().mean().sort_values(ascending=False)*100

null_values_train = null_values_train[null_values_train>0]
null_values_test = test_df.isnull().mean().sort_values(ascending=False)*100

null_values_test = null_values_test[null_values_test>0]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

sns.barplot(x=null_values_train.index, y=null_values_train.values, ax=ax[0]).set_title('Null values pourcentage in train set')

sns.barplot(x=null_values_train.index, y=null_values_test.values, ax=ax[1]).set_title('Null values pourcentage in test set')

plt.show()
X_train = train_df[[col for col in train_df.columns if col!="MEDV"]]

y_train = train_df["MEDV"]



X_test = test_df[[col for col in test_df.columns if col!="MEDV"]]

t_test = test_df["MEDV"]
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
train_df_dropped_na = train_df.dropna()

print("The number of rows in the train_df_dropped_na is {}".format(train_df_dropped_na.shape[0]))
N = 3000

sex = np.random.choice(["Male", "Female"], N, p=[0.6, 0.4])

height = 140 + (200-140) * np.random.rand(N)

weight = 40 + (120-40) * np.random.rand(N)

salary = 30000+(80000-30000) * np.random.rand(N)

df = pd.DataFrame(data=[sex, height, weight, salary]).transpose()

df.columns = ["Sex", "Height", "weight", "salary"]
# Initialize the Dice columns

df["Dice"] = df["Sex"]

# Fill the Dice column with the probability values

df["Dice"] = np.random.choice([1, 2, 3, 4, 5, 6], N, p=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

# Dtermine indices where Dice=6

index = df[df["Dice"]==6].index

# Replace with NaN

df.loc[index,"Height"] = np.nan
# Initialize the Dice columns

df["Height_missing"] = df["Height"]

# The column is false

df["Height_missing"] = False

# Replace where Height_missing with True where Height is missing

df.loc[df[df['Height'].isnull()].index, "Height_missing"] = True
df[df["Height_missing"]==True].groupby("Sex")["Height_missing"].count()
df[df["Height_missing"]==False].groupby("Sex")["Height_missing"].count()
#          |   True   |   False  |



# Female   |   197    |   1025   |



# Male     |   307    |   1471   |
table = [[197, 1025],[307,1471]]
from scipy.stats import chi2_contingency

chi2, p, dof, ex = chi2_contingency(table)
print("The p-value is esqual to {}".format(p))