# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import string as st



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

original =  pd.read_csv('../input/data.csv')

fi=original

fi=pd.DataFrame(fi)

fi.head()
fi = fi.drop(columns='Unnamed: 0')

fi = fi.drop(columns='ID')

fi = fi.drop(columns='Photo')

fi = fi.drop(columns='Flag')

fi = fi.drop(columns='Club Logo')

fi = fi.drop(columns='Joined')
fi['Value'].unique()
#Correct currencies

curs=["Release Clause", "Value", "Wage"]

for cur in curs:

    

    def curr_value(x):

        x = str(x).replace('€', '')

        if('M' in str(x)):

            x = str(x).replace('M', '')

            x = float(x) * 1000000

        elif('K' in str(x)):

            x = str(x).replace('K', '')

            x = float(x) * 1000

        return float(x)

    fi[cur] = fi[cur].apply(curr_value)
#Correct -Dismiss + values

cols=["LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW","LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM","CDM", "RDM", "RWB", "LB", "LCB", "CB", "RCB", "RB"]

for col in cols:

    fi[col]=fi[col].str[:-2]

    fi[col]=fi[col].astype(float)
fi['Contract Valid Until']=fi['Contract Valid Until'].str[-4:]

fi['Contract Valid Until']=fi['Contract Valid Until'].astype(float)
#Corect height values 

fi['Height']=fi['Height'].str.replace("'",'.')

fi['Height']=fi['Height'].astype(float)



#Correct Weight

fi['Weight']=fi['Weight'].str[:-3]

fi['Weight']=fi['Weight'].astype(float)
#X and y assignments

X = fi.loc[:, fi.columns != 'Value']

y=fi.loc[:,['Value']]

X = X.drop(columns='Name')

X = X.drop(columns='Real Face')

X.dtypes
obj_df = X.select_dtypes(include=['object']).copy()

#Encoding &missing Values

cols=obj_df.columns

for col in cols:

    print(col)

    X[col].replace(np.NaN,'NotAv',inplace=True)

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    labelencoder_X = LabelEncoder()

    X[col] = labelencoder_X.fit_transform(X[col])
#See Correlations



X1 = pd.DataFrame(X)

corr = X1.corr()

fig = plt.figure(figsize=(50,20))

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(X1.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(X1.columns)

ax.set_yticklabels(X1.columns)

plt.show()
#drop highly correlated attributes

corr_matrix = X1.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

X=X.drop(columns=to_drop, axis=1)

print(to_drop)