import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from tensorflow.keras.layers import Input,Dense

from tensorflow.keras.models import Sequential

from keras.models import Model

import tensorflow as tf

%matplotlib inline

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/ny-property-data/NY property data.csv')

df.head()
df = df[['BBLE','FULLVAL', 

         'AVLAND', 

         'AVTOT', 

         'LTFRONT', 

         'LTDEPTH', 

         'BLDFRONT', 

         'BLDDEPTH', 

         'STORIES', 

         'ZIP', 

         'TAXCLASS', 

         'B',

         'BLOCK',

         'BLDGCL']]

df['BLOCK'] = df['BLOCK'].astype('category')

df['BLDGCL'] = df['BLDGCL'].astype('category')

#combine 0 and NA

df.replace(0, np.nan, inplace=True)
#fill ZIP nan

df['ZIP']=df.groupby(['B','BLOCK'])['ZIP'].transform(lambda x: x.fillna(x.median()))

df['ZIP']=df.groupby(['B'])['ZIP'].transform(lambda x: x.fillna(x.median()))

a=df['ZIP'].isnull().sum()

#fill NAs: FULLVAL, AVLAND, AVTOT

HV=['FULLVAL', 'AVLAND', 'AVTOT']

for i in HV:

    df[i]=df.groupby(['BLDGCL'])[i].transform(lambda x: x.fillna(x.median()) if len(x)>=5 else x)

    df[i]=df.groupby(['B'])[i].transform(lambda x: x.fillna(x.median()))

b=df[HV].isnull().sum()

#Filling NAs: LTFRONT, LTDEPTH,BLDFRONT, BLDDEPTH

HP=['LTFRONT', 'LTDEPTH', 'BLDFRONT', 'BLDDEPTH']

for hpi in HP:

    df[hpi]=df.groupby('BLDGCL')[hpi].transform(lambda x: x.fillna(x.median()))

    df[hpi]=df.groupby('B')[hpi].transform(lambda x: x.fillna(x.median()))



df[HP].isnull().sum()

c=df[HP].isnull().sum()

# Filling NAs: STORIES

df['STORIES'] = df.groupby('BLDGCL')['STORIES'].transform(lambda x: x.fillna(x.median()))

df['STORIES'] = df.groupby('B')['STORIES'].transform(lambda x: x.fillna(x.median()))

d=df['STORIES'].isnull().sum()

print(a,b,c,d)
from IPython.display import Image

Image("../input/create-variables/Screen Shot 2019-02-22 at 10.24.17 PM.png")
df['lotarea']=df['LTFRONT']*df['LTFRONT']

df['bldarea']=df['BLDFRONT']*df['BLDDEPTH']

df['bldvol']=df['bldarea']*df['STORIES']



new_cols=['lotarea','bldarea','bldvol']

norm_vols=['FULLVAL','AVLAND', 'AVTOT']



for n in norm_vols:

    for j in new_cols:

        df[n+'/'+j]=df[n]/df[j]

# create variables

df['zip3'] = df['ZIP']//100

df['zip5'] = df['ZIP']//1



#scale value

scale_value=['FULLVAL/lotarea',

 'FULLVAL/bldarea',

 'FULLVAL/bldvol',

 'AVLAND/lotarea',

 'AVLAND/bldarea',

 'AVLAND/bldvol',

 'AVTOT/lotarea',

 'AVTOT/bldarea',

 'AVTOT/bldvol',]

scale_facter=['zip3','zip5','TAXCLASS','B']

for i in scale_value:

    df[i+'_scale by_all']=df[i]/df[i].mean()

    for j in scale_facter:

        df[i+'_scale by_'+j]=df.groupby(j)[i].apply(lambda x: x/(x.mean()))

#clean process data

df=df.drop(['FULLVAL',

 'AVLAND',

 'AVTOT',

 'LTFRONT',

 'LTDEPTH',

 'BLDFRONT',

 'BLDDEPTH',

 'STORIES',

 'ZIP',

 'TAXCLASS',

 'B',

 'BLOCK',

 'BLDGCL',

 'lotarea',

 'bldarea',

 'bldvol',

 'FULLVAL/lotarea',

 'FULLVAL/bldarea',

 'FULLVAL/bldvol',

 'AVLAND/lotarea',

 'AVLAND/bldarea',

 'AVLAND/bldvol',

 'AVTOT/lotarea',

 'AVTOT/bldarea',

 'AVTOT/bldvol',

 'zip3',

 'zip5'], axis=1)

df.head()
df.shape
feature_values=df.columns.values.tolist()

feature_values.pop(0)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def scaleColumns(df, feature_values):

    for col in feature_values:

        df[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(df[col])),columns=[col])

    return df

scaled_df = scaleColumns(df,feature_values)
scaled_df.head()
scaled_df.set_index('BBLE', inplace=True)

scaled_df.head()
from sklearn.decomposition import PCA

npc=20

pca = PCA(n_components=npc)

principalComponents = pca.fit_transform(scaled_df[feature_values].values)

pca.get_covariance()

explained_variance=pca.explained_variance_ratio_

explained_variance

X=list(range(1,npc+1))

plt.bar(X,explained_variance,label='individual explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')
values = list(explained_variance)

total  = 0

sums   = []



for v in values:

  total = total + v

  sums.append(total)

sums
npc=8

pca = PCA(n_components=npc)

principalComponents = pca.fit_transform(scaled_df.values)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 

                          'principal component 2', 

                          'principal component 3', 

                          'principal component 4', 

                          'principal component 5', 

                          'principal component 6', 

                          'principal component 7', 

                          'principal component 8'],index=scaled_df.index)

principalDf['BBLE']=principalDf.index

principalDf.reset_index(drop=True, inplace=True)

principalDf.head()
feature_values2=principalDf.columns.values.tolist()

feature_values2.pop(-1)
scaled_principalDf= scaleColumns(principalDf,feature_values2)
scaled_principalDf.head()
Image("../input/score-image/score1.png",width=200, height=600)
n=8

scaled_principalDf['zscore']=0

for pc in feature_values2:

    scaled_principalDf['zscore']=scaled_principalDf['zscore']+(scaled_principalDf[pc])**n

scaled_principalDf['zscore']=(scaled_principalDf['zscore'])**(1/n)

scaled_principalDf.head()
Image("../input/score-image/score2.2.png",width=300, height=800)
Image("../input/score-image/score2.1.png",width=200, height=600)
from keras.models import Model, load_model

from keras.layers import Input, Dense

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import regularizers

input_data=scaled_principalDf[feature_values2]

input_dim = 8

encoding_dim = 4



input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation="tanh", 

                activity_regularizer=regularizers.l1(10e-5))(input_layer)

encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)

decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
nb_epoch = 1

#batch_size = 

autoencoder.compile(optimizer='adam', 

                    loss='mean_squared_error', 

                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",

                               verbose=0,

                               save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs',

                          histogram_freq=0,

                          write_graph=True,

                          write_images=True)

history = autoencoder.fit(input_data, input_data,

                    epochs=nb_epoch,

                    shuffle=True,

                    verbose=1,

                    callbacks=[checkpointer, tensorboard]).history

predictions = autoencoder.predict(input_data)
autoencoded_data = pd.DataFrame(predictions,columns=['enPC1','enPC2','enPC3','enPC4','enPC5','enPC6','enPC7','enPC8'])

autoencoded_data.head()
n=8

for i in range(n):

    scaled_principalDf[i]=abs(autoencoded_data.iloc[:,i]-scaled_principalDf.iloc[:,i])

#data.iloc[:,1] # second column of data frame (last_name)
scaled_principalDf.head()
n=8

diff=list(range(n))

scaled_principalDf['autoencoder']=0

for pc in diff:

    scaled_principalDf['autoencoder']=scaled_principalDf['autoencoder']+(scaled_principalDf[pc])**n

scaled_principalDf['autoencoder']=(scaled_principalDf['autoencoder'])**(1/n)
scaled_principalDf.head()
df=scaled_principalDf

df['Zscore_Rank'] = df['zscore'].rank(ascending=True)

df['Autoencoder_Rank'] = df['autoencoder'].rank(ascending=True)

df_final=df[['BBLE','zscore','autoencoder','Zscore_Rank','Autoencoder_Rank']]

df_final['Final_Score']=df_final['zscore']+df_final['autoencoder']

df_final['Final_Rank']=0.7*df_final['Autoencoder_Rank'] +0.5*df_final['Zscore_Rank']

df_final = df_final.sort_values(by=['Final_Rank'], ascending=False)

df_final.head(10)
fraud=df_final['BBLE'].tolist()[:10]

df.loc[df['BBLE'].isin(fraud)]
import matplotlib

import matplotlib.pyplot as plt

import numpy as np



#xs = np.random.normal(size=int(1e6))

fig, ax = plt.subplots(1, 3, figsize=(16,5))

ax[0].hist(df_final['Final_Score'], bins=30)

ax[0].set_yscale('log')

ax[1].hist(df_final['zscore'], bins=30)

ax[1].set_yscale('log')

ax[2].hist(df_final['autoencoder'], bins=30)

ax[2].set_yscale('log')
