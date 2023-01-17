import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense,Dropout
pd
df=pd.read_csv('../input/train.csv')
df.head()
X_train=df.iloc[:,1:].values
y_train=df.iloc[:,0].values
X_test=pd.read_csv('../input/test.csv').values
print(X_train.shape)

X_test.shape
y_train=y_train.reshape(42000,1)
X_train=X_train.reshape(X_train.shape[0],28,28)
plt.imshow(X_train[0],cmap='binary')

plt.title(y_train[0])
X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_test=X_test.reshape(X_test.shape[0],28,28,1)
mean=X_train.mean().astype(np.float32)

std=X_train.std().astype(np.float32)
mean
def standardise(x):

    return (x-mean)/std
y_train.shape
from keras.utils.np_utils import to_categorical

y_train=to_categorical(y_train)
X_train.shape
y_train.shape
from keras.layers.core import Lambda,Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

gen=ImageDataGenerator()
y_train.shape
from sklearn.model_selection import train_test_split

X=X_train

y=y_train

X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.10,random_state=42)

batches=gen.flow(X_train,y_train,batch_size=64)

val_batches = gen.flow(X_val,y_val,batch_size=64)

from keras.layers import Convolution2D , MaxPooling2D
from keras.layers.normalization import BatchNormalization
def batch_model():

    model=Sequential([

        Lambda(standardise,input_shape=(28,28,1)),

        Convolution2D(32,(3,3),activation='relu'),

        BatchNormalization(axis=1),

        Convolution2D(32,(3,3),activation='relu'),

        MaxPooling2D(),

        BatchNormalization(axis=1),

        Convolution2D(64,(3,3),activation='relu'),

        BatchNormalization(axis=1),

        Convolution2D(64,(3,3),activation='relu'),

        MaxPooling2D(),

        Flatten(),

        BatchNormalization(axis=1),

        Dense(512,activation='relu'),

        BatchNormalization(axis=1),

        Dense(10,activation='softmax')])

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model   
complete=batch_model()

complete.optimizer.lr=0.001
from keras.preprocessing.image import ImageDataGenerator
gen=ImageDataGenerator()

batches=gen.flow(X,y,batch_size=64)
compl_hist = complete.fit_generator(generator=batches,steps_per_epoch=batches.n,epochs=5)
predictions = complete.predict_classes(X_test,verbose=0)

submissions = pd.DataFrame({'ImageID':list(range(1,len(predictions)+1)),'Label':predictions})

submissions.to_csv('digits.csv',index=False,header=True)
df=pd.read_csv('digits.csv')
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(df)