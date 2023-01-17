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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

import string

import category_encoders as ce

import time
rawtrain=pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

train=rawtrain.drop('id',axis=1)



#======encode ordinal

cate_ord=['ord_1','ord_2']

for c in cate_ord:

    print(rawtrain[c].unique())

levelmap={c:i for i,c in enumerate(['Novice','Contributor', 'Expert', 'Master','Grandmaster'])}

train['ord_1']=train['ord_1'].replace(levelmap)

tempratmap={c:i for i,c in enumerate(['Freezing','Cold', 'Warm','Hot' , 'Boiling Hot' ,'Lava Hot' ])}

train['ord_2']=train['ord_2'].replace(tempratmap)

lowermap={c:i for i,c in enumerate(string.ascii_lowercase)}

train['ord_3']=train['ord_3'].replace(lowermap)

upperletter=rawtrain['ord_4'].unique().tolist()

upperletter.remove(np.nan)

upperletter.sort()

uppermap={c:i for i,c in enumerate(string.ascii_uppercase)}

train['ord_4']=train['ord_4'].replace(uppermap)

#/ord_5

alletter=string.ascii_letters

allmap={c:i for i,c in enumerate(alletter)}

def getP(x,p):

    if pd.isnull(x):

        return x

    else:

        if p==0:

            return x[0]

        else:

            return x[1]

        

train['ord_5_0']=rawtrain['ord_5'].apply(lambda x: getP(x,0)).replace(allmap)

train['ord_5_1']=rawtrain['ord_5'].apply(lambda x: getP(x,1)).replace(allmap)

train=train.drop('ord_5',axis=1)

#======encode binary and nominal+label to num for k mode clustering:https://www.kaggle.com/teejmahal20/clustering-categorical-data-k-modes-cat-ii

normcol59=['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

train_cluster=train.drop(normcol59,axis=1)

for c in train_cluster.columns:

    train_cluster[c].fillna(train_cluster[c].mode()[0], inplace = True)



bincol_labeled=['bin_3', 'bin_4']

binOE=OrdinalEncoder()

train_cluster[bincol_labeled]=binOE.fit_transform(train_cluster[bincol_labeled])



normcol_labeled=['nom_0','nom_1','nom_2', 'nom_3', 'nom_4']

binOE=OrdinalEncoder()

train_cluster[normcol_labeled]=binOE.fit_transform(train_cluster[normcol_labeled])
#==========test independency

import scipy.stats as scs



def chi_square_of_df_cols(df, col1, col2):

    df_col1, df_col2 = df[col1], df[col2]



    result = [[sum((df_col1 == cat1) & (df_col2 == cat2))

               for cat2 in df_col2.unique()]

              for cat1 in df_col1.unique()]



    return scs.chi2_contingency(result)



chi_matrix=np.zeros([len(train_cluster.columns),len(train_cluster.columns)])

for i,r in enumerate(train_cluster.columns):

    for j,c in enumerate(train_cluster.columns):

        print('{}{}'.format(i,j),flush=True)

        if i!=j:

            stemp,tp,_,_=chi_square_of_df_cols(train_cluster, r, c)

            chi_matrix[i,j]=tp

for i,r in enumerate(train_cluster.columns):

    for j,c in enumerate(train_cluster.columns):

        if i==j:

            chi_matrix[i,j]=np.nan

f, ax = plt.subplots(figsize=(10, 10))

colormap = plt.cm.Greens_r

sns.heatmap(pd.DataFrame(chi_matrix,columns=train_cluster.columns,index=train_cluster.columns), 

             cmap=colormap, square=True, linewidths=.5)