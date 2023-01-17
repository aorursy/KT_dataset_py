import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from keras.regularizers import l2

from keras.utils import to_categorical

import keras.backend as K

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint , ReduceLROnPlateau

from copy import copy

from sklearn.metrics import mean_absolute_error

from keras.layers import Dense, Dropout , BatchNormalization

from keras.models import Sequential

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler , MinMaxScaler

from imblearn.over_sampling import SMOTE , ADASYN , BorderlineSMOTE , RandomOverSampler , SVMSMOTE , SMOTENC

from keras import optimizers

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
# from google.colab import drive

# drive.mount('/content/drive')
def pre_proc_tr(df2) :

    ind = []

    for col in df2.columns :

      for i in range(df2.shape[0]) :

        if df2[col][i] == "?" :

          ind.append(i) 

    df2.drop(df2.index[ind], inplace=True)

    df2.reset_index(inplace = True,drop = True) 

    df2["Small"] = [1.0 if df2["Size"][x] == "Small" else 0.0 for x in range(len(df2["Size"]))]

    df2["Medium"] = [1.0 if df2["Size"][x] == "Medium" else 0.0 for x in range(len(df2["Size"]))]

    df2["Big"] = [1.0 if df2["Size"][x] == "Big" else 0.0 for x in range(len(df2["Size"]))] 

    df2 = df2.drop(columns = ["Size"])

    df2 = df2.astype(np.float64)



    return df2

def pre_proc(df2,sca="") :

    

    

    # x_col = [x for x in df2.columns if (x != "Class" and x != "ID" )]

    # sm = SVMSMOTE(random_state=1)

    # X, Y = sm.fit_sample(df2[x_col],df2["Class"])

    # print(X.shape,Y.shape)

    # x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size = .3,random_state=1,shuffle=True)

    # print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)



    x_col = [x for x in df2.columns if (x != "Class" and x != "ID" )]

    x_train , x_test , y_train , y_test = train_test_split(df2[x_col],df2["Class"],test_size = .3,random_state=1,shuffle=True)

    print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)



    x_temp , y_temp = x_train.values , y_train.values

    print(x_temp.shape,y_temp.shape)

    tp = [ SVMSMOTE ]

    for t in tp :

        print(str(t))

        sm = t(random_state=1)

        x_val, y_val = sm.fit_sample(x_train,y_train)

        x_val = x_val.values

        y_val = y_val.values.reshape((-1,1))

        x_temp = np.vstack((x_temp,x_val))

        y_temp = np.vstack((y_temp.reshape((-1,1)),y_val))

    x_train , y_train = x_temp , y_temp

    print(x_train.shape,y_train.shape)







#     ss = Mi().fit(x_train)

#     x_train = ss.transform(x_train)

#     x_test = ss.transform(x_test)



    

    print(np.unique(y_test,return_counts=True))

    print(np.unique(y_train,return_counts=True))



    y_train = to_categorical(y_train)

    print(y_train.shape)

    return x_train , y_train , x_test , y_test 
df2 = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", sep=',')

df2=pre_proc_tr(df2)

x_val = df2[[x for x in df2.columns if (x != "Class" and x != "ID" )]]

y_val = to_categorical(df2["Class"])

x_train , y_train , x_test , y_test = pre_proc(df2,"tra")

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

def get_model1 () :

    model = Sequential()

    model.add(Dense(88,input_dim=13, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(Dense(44, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(Dense(22, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization()) 

    model.add(Dense(11, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    model.add(Dense(6,activation='softmax'))

    

    adam = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

    print(model.summary())

    print("model 1 starts.")

    return model
def get_model2 () :

    model = Sequential()

    model.add(Dense(88,input_dim=13, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(Dense(44, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(Dense(22, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization()) 

    model.add(Dense(11, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    model.add(Dense(6,activation='softmax'))

    

    adam = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

    print(model.summary())

    print("model 2 starts.")

    

    return model
def get_model3 () :

    model = Sequential()

    model.add(Dense(88,input_dim=13, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(Dense(44, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(Dense(22, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization()) 

    model.add(Dense(11, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    model.add(Dense(6,activation='softmax'))

    

    adam = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

    print(model.summary())

    print("model 3 starts.")

    

    return model
def get_model4 () :

    model = Sequential()

    model.add(Dense(88,input_dim=13, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(Dense(44, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(Dense(22, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization()) 

    model.add(Dense(11, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    model.add(Dense(6,activation='softmax'))

    

    

    adam = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

    print(model.summary())

    print("model 4 starts.")

    

    return model
def get_model5 () :

    model = Sequential()

    model.add(Dense(88,input_dim=13, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(Dense(44, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(Dense(22, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    #model.add(Dropout(0.2))

    model.add(BatchNormalization()) 

    model.add(Dense(11, activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    model.add(Dense(6,activation='softmax'))

    

    adam = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

    print("model 5 starts.")

    

    print(model.summary())

    return model
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',np.unique(np.argmax(y_train,axis = 1)),np.argmax(y_train,axis = 1))
models = [get_model1]

i = 1

callbacks = [#ModelCheckpoint(filepath='/content/drive/My Drive/NNFL/Lab1/best_model'+str(i)+'.h5', monitor='val_loss', save_best_only=True),

          #EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None, restore_best_weights=False),

          ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)]

model = get_model1()

history = model.fit(x_train,y_train,epochs=1000,validation_split = .2 ,class_weight=class_weights,callbacks=callbacks,batch_size=16)

i +=1
pred = model.predict(x_test)

print("model "+str(i)+" test acc : " + str(sum(np.argmax(pred,axis = 1) == y_test)/pred.shape[0]))

pred = model.predict(x_train)

print("model "+str(i)+"train acc : " + str(sum(np.argmax(pred,axis = 1) == np.argmax(y_train,axis = 1))/pred.shape[0])) 

pred = model.predict(x_val)

print("model "+str(i)+"actual acc : " + str(sum(np.argmax(pred,axis = 1) == np.argmax(y_val,axis = 1))/pred.shape[0]))

print()
df = pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')

df1 = pd.read_csv('/kaggle/input/bitsf312-lab1/sample_submission.csv')
df["Small"] = [1.0 if df["Size"][x] == "Small" else 0.0 for x in range(len(df["Size"]))]

df["Medium"] = [1.0 if df["Size"][x] == "Medium" else 0.0 for x in range(len(df["Size"]))]

df["Big"] = [1.0 if df["Size"][x] == "Big" else 0.0 for x in range(len(df["Size"]))] 

df = df.drop(columns = ["Size"])
p1 = None

for i in range(1,2) :

    if i == 1 :

        p1= model.predict(df[[x for x in df.columns if (x != "Class" and x != "ID" )]])

p1 += model.predict(df[[x for x in df.columns if (x != "Class" and x != "ID" )]])

p1 = p1/6

np.argmax(p1,axis = 1)
df1["Class"] = np.argmax(p1,axis = 1)

df1.to_csv("sub_fin.csv",index = False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df1)