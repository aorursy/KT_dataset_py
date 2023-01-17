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
from IPython.display import Image

import os

!ls ../input/
Image("../input/mlpoke/MLPOke.png")
import pandas as pd

import numpy as np

from category_encoders import *





data={'Type':['Fire','Water','Bug', 'Fire', 'Fire','Bug','Water','Bug','Ice'],'Height':['Short','Normal','Very short','Tall','Normal','Short','Tall','Very short','Tall'],'Stats_total':[495,525,195,580, 525,500,670,405,580],'Legendary':[0,0,0,1,0,0,1,0,1]}

df_main=pd.DataFrame(data)

df_main
df=df_main.copy()

height_dict ={'Very short':1, 'Short':2, 'Normal':3, 'Tall':4}

df['Ordinal_Height']=df.Height.map(height_dict)

df[['Height','Ordinal_Height']]
df=df_main.copy()

df_Height=pd.get_dummies(df[['Height']],prefix='T')



pd.concat([df[['Height']],df_Height],axis=1).head()
from sklearn.preprocessing import OneHotEncoder

df=df_main.copy()



ohe=OneHotEncoder()

ohe=ohe.fit_transform(df[['Height']]).toarray()

newdata=pd.DataFrame(ohe)



dfh=pd.concat([df[['Height']],newdata],axis=1)

dfh.head()
from category_encoders import BinaryEncoder

df=df_main.copy()



be=BinaryEncoder(cols=['Type'])

newdata=be.fit_transform(df['Type'])



EncounterStep= pd.DataFrame([1,2,3,1,1,3,2,3,4],columns=["EncounterStep"]) #Test it your self if this correct

dfh=pd.concat([df[['Type']],EncounterStep,newdata],axis=1)

dfh
df=df_main.copy()



dfTemp=df.groupby("Type").size()/len(df) #Group it by type, find the size of each type, and divide by total event

df['Type_freq']=df['Type'].map(dfTemp) #dfTemp is a dataframe type



pd.concat([df[['Type']],df['Type_freq']],axis=1)
from sklearn.feature_extraction import FeatureHasher

df=df_main.copy()



fg = FeatureHasher(n_features=2, input_type='string')

hashed_features = fg.fit_transform(df['Type'])

hashed_features = hashed_features.toarray()



df=pd.concat([df[['Type']], pd.DataFrame(hashed_features)], axis=1)

df
from category_encoders import HelmertEncoder

df=df_main.sample(5)



he=HelmertEncoder(cols=['Height'])

newcolumn=he.fit_transform(df['Height'])



df=pd.concat([df[['Height']],newcolumn],axis=1)

df
from category_encoders import BackwardDifferenceEncoder

df=df_main.sample(5)



bwde = BackwardDifferenceEncoder()

newcolumns=bwde.fit_transform(df['Type'])



pd.concat([df[['Type']],newcolumns],axis=1)
df=df_main.copy()



mean_encode=df.groupby("Type")['Legendary'].mean()

df['Type_legendary']=df['Type'].map(mean_encode)



df[['Type','Legendary','Type_legendary']]
from category_encoders import TargetEncoder

df=df_main.copy()



TE = TargetEncoder(cols=['Type'])

df['Type_legendary']=TE.fit_transform(df['Type'],df['Legendary'])



df[['Type','Legendary','Type_legendary']]
from category_encoders import LeaveOneOutEncoder

df=df_main.copy()



LOOE = LeaveOneOutEncoder(cols=['Type'], sigma=0.2)

df['Type_legendary']=LOOE.fit_transform(df['Type'], df['Legendary'])



df[['Type','Legendary','Type_legendary']]
from category_encoders import WOEEncoder



WOEE = WOEEncoder(cols=['Type'],regularization=0.5)

df['Type_legendary']=WOEE.fit_transform(df['Type'], df['Legendary'])



df[['Type','Legendary','Type_legendary']]
from category_encoders import JamesSteinEncoder



JSE= JamesSteinEncoder(sigma=0.1)

newcolumns=JSE.fit_transform(df['Type'], df['Legendary'])



df['JSE_col']=newcolumns

df[['Type','Legendary','JSE_col']]
from category_encoders import MEstimateEncoder

df=df_main.copy()



MEE=MEstimateEncoder(m=2)

newcolumns = MEE.fit_transform(df['Type'], df['Legendary'])



df['MEE_col']=newcolumns

df[['Type','Legendary','MEE_col']]
Image("../input/somepictures/en_dis.png")