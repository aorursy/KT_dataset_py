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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns',None)
df =pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df.info()
df.shape
df.isnull().sum()
na_features = [features for features in df.columns if df[features].isnull().sum()>1]
for feature in na_features:

    print(feature, np.round(df[feature].isnull().mean(),4),'% missing values')
for feature in na_features:

    data = df.copy()

    

    data[feature] = np.where(df[feature].isnull(),1,0)

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.title(feature)

    plt.show()
print("Id of House is {}".format(len(df['Id'])))
numerical_ft = [ft for ft in df.columns if df[ft].dtypes !='O']

print("Total numerical features = {}".format(len(numerical_ft)))
df[numerical_ft].head()
year_feature = [feature for feature in numerical_ft if 'Yr' in feature or 'Year' in feature]

year_feature
for feature in year_feature:

    print(feature, df[feature].unique())
df.groupby('YrSold')['SalePrice'].median().plot()

plt.xlabel('Year Sold')

plt.ylabel('Median House Price')

plt.title('House Price vs. Year Sold')
for feature in year_feature:

    if feature !='YrSold':

        data = df.copy()

        data[feature] = data['YrSold'] - data[feature]

        plt.scatter(data[feature],data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('SalePrice')

        plt.show()
discrete_feature=[feature for feature in numerical_ft if len(df[feature].unique())<25 and feature not in year_feature+['Id']]

print("Discrete Variables Count: {}".format(len(discrete_feature)))
discrete_feature
df[discrete_feature].head()
for ft in discrete_feature:

    data = df.copy()

    data.groupby(ft)['SalePrice'].median().plot.bar()

    plt.xlabel(ft)

    plt.ylabel('Median House Price')

    plt.title(ft)

    plt.show()
continuous_feature=[feature for feature in numerical_ft if feature not in discrete_feature+year_feature+['Id']]

print("Continuous feature Count {}".format(len(continuous_feature)))
for feature in continuous_feature:

    data=df.copy()

    data[feature].hist(bins=25)

    plt.xlabel(feature)

    plt.ylabel("Count")

    plt.title(feature)

    plt.show()