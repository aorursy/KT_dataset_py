# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

joined = df_train.append(df_test)
df_train.head(3)
cont_vars = ['Age', 'Fare', 'CabinNumber', 'TicketNumber'] # continuous variables

cat_vars = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Embarked', 'CabinClass'] # categorical variables

xx_vars = ['Name', 'TicketType'] # weird textish variables

dep_var = 'Survived' # dependent variable
def augment_df(df):

    df['CabinClass'] = df.Cabin.str.extract(r'(\D+)')

    df['CabinNumber'] = df.Cabin.str.extract(r'(\d+)').astype(float)

    df['TicketType'] = df.Ticket.str.extract(r'(\D+)') # NEEDS WORK

    df['TicketNumber'] = df.Ticket.str.extract(r'(\d+)').astype(float) # NEEDS WORK

    for col in cont_vars:

        df[col] = df[col].fillna(df_train[col].mean())

    for col in cat_vars:

        df[col] = df[col].fillna('XXUNKNOWN')

#     for col in xx_vars:

#         WTF do we do here?

    return df

aug_df = augment_df(df_train)

aug_df.head()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder
linreg = LinearRegression()

X = df_train.loc[:, df_train.columns != 'Survived']

Y = df_train['Survived']



train, valid = train_test_split(X, test_size=0.2)
for col in cat_vars:

    labelencoder = LabelEncoder()

    x[:, 0] = labelencoder.fit_transform(x[:, 0])



    onehotencoder = OneHotEncoder(categorical_features = [0])

    x = onehotencoder.fit_transform(x).toarray()

ohe = OneHotEncoder(sparse=False)

X = ohe.fit_transform(X)

X
# linreg.fit(X,Y)