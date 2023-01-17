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
from keras.models import Sequential

from keras.layers import LSTM,Dense,Dropout

from keras.models import load_model, Model

from sklearn.model_selection import train_test_split



import pandas as pd

import numpy as np
df = pd.read_csv("/kaggle/input/predict-volcanic-eruptions-ingv-oe/train/416906269.csv")

df.fillna(0, inplace=True)

print(df)

df = pd.read_csv("/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv")

df.fillna(0, inplace=True)

print(df)

# train_x = 

list1 = []

for i, j in zip(df["segment_id"].unique().tolist(), df["time_to_eruption"].unique().tolist()):

    if j!=0:

        dftemp = pd.read_csv("/kaggle/input/predict-volcanic-eruptions-ingv-oe/train/%s.csv"%i)

        dftemp.fillna(0, inplace=True)

        total_data = np.concatenate((dftemp.abs().mean().to_numpy(),

                                     dftemp.std().to_numpy(),

                                     dftemp.mean().to_numpy(),

                                     dftemp.var().to_numpy(),

                                     dftemp.min().to_numpy(),

                                     dftemp.max().to_numpy(),

                                     dftemp.median().to_numpy(),

                                     dftemp.quantile([0.1,0.25,0.5,0.75,0.9]).to_numpy().reshape(1,-1)[0])).tolist()

        print(total_data)



        print(i)

        list1.append(total_data)

from keras.layers import Input, Dense

from keras.models import Model

from keras.layers.core import Flatten, Dense, Dropout, Activation

from keras.layers.normalization import  BatchNormalization  as bn

from  keras.layers.pooling import MaxPooling1D as pool



import  keras.utils 



from sklearn.model_selection import train_test_split



# print(list1)

# print(df["time_to_eruption"].tolist())

train_x, test_x, train_y, test_y = train_test_split(list1,

                                    df["time_to_eruption"].tolist(),

                                    test_size=0.1,

                                    shuffle=True)

# This returns a tensor

inputs = Input(shape=(np.array(train_x).shape[1],))

# x = bn()(x)

# del x

# a layer instance is callable on a tensor, and returns a tensor

x = Dense(1000, activation='relu')(inputs)

# x = bn()(x)

# x = Activation('relu')(x)



# x = Activation('sigmoid')(x)

# x = Dense(1000)(x)





# x = Dense(128)(x)

# x = bn()(x)

x = Dropout(0.7)(x)

# x = Dense(64)(x)

# x = bn()(x)

# x = Activation('relu')(x)

# x = Dense(32)(x)

# x = bn()(x)

# x = Activation('linear')(x)

predictions = Dense(1, activation='relu')(x)

# predictions = Dense(10, activation='softmax')(x)



# This creates a model that includes

# the Input layer and three Dense layers

model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='adam',

              loss='mean_absolute_error',

             metrics=['mae'])

# model.compile(optimizer='rmsprop',

#               loss='sparse_categorical_crossentropy',

#              metrics=['acc'])

model.fit(train_x,train_y, validation_data=(test_x, test_y)

          ,batch_size=8,epochs=600)
%%time

sample_submission_df=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')

n_f=12

total_data_test_=np.empty((sample_submission_df.shape[0],n_f*10))

for i_,seg_ in enumerate(sample_submission_df['segment_id']):

    the_df=pd.read_csv(f'/kaggle/input/predict-volcanic-eruptions-ingv-oe/test/{seg_}.csv').fillna(0)

    total_data_test_[i_,:]=np.concatenate((the_df.abs().mean().to_numpy(),

                                    the_df.std().to_numpy(),

                                    the_df.mean().to_numpy(),

                                    the_df.var().to_numpy(),

                                    the_df.min().to_numpy(),

                                    the_df.max().to_numpy(),

                                    the_df.median().to_numpy(),

                                    the_df.quantile([0.1,0.25,0.5,0.75,0.9]).to_numpy().reshape(1,-1)[0]))
sample_submission_df1=pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')

sample_submission_df['time_to_eruption']=model.predict(total_data_test_)

sample_submission_df.to_csv('/kaggle/working/submission.csv',index=False)
the_df=pd.read_csv(f'/kaggle/input/predict-volcanic-eruptions-ingv-oe/test/{1001028887}.csv').fillna(0)

print(the_df)