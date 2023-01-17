# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import pickle 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

def savepickle(obj,name):

    with open(name+'.pkl','wb') as k:

        pickle.dump(obj,k)





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout,Activation,Conv2D,MaxPooling2D,Flatten

from tensorflow.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

import kerastuner as kt

import cv2

import numpy as np

import zipfile 

import datetime, os

import time

os.getcwd()
print(os.listdir('../input'))



with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:

    z.extractall(".")



with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip","r") as z:

    z.extractall(".")
main_dir = "/kaggle/working/"

train_dir = "train"

path = os.path.join(main_dir,train_dir)
X = []

y = []

convert = lambda category : int(category == 'dog')

def create_test_data(path):

    for p in os.listdir(path):

        category = p.split(".")[0]

        category = convert(category)

        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

        new_img_array = cv2.resize(img_array, dsize=(80, 80))

        X.append(new_img_array)

        y.append(category)

    
create_test_data(path)

X = np.array(X).reshape(-1, 80,80,1)

y = np.array(y)



X = X/255

X=X[:128]

y=y[:128]
        



def bui_mod(hp):

    NAME="cats_vs_dogs_{}".format(int(time.time()))

    tensorboard=TensorBoard(log_dir='logs\{}'.format(NAME))



    model=Sequential()                                              ######Start creating the model 

    model.add(Conv2D(hp.Int('Filter_per_conv',64,256,64),(3,3),input_shape=X.shape[1:],padding='same'))      #####Conv2D network

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))



    for covs in range(hp.Int('num_of_layers',1,4,1)):

        model.add(Conv2D(hp.Int('Filter_per_conv',64,256,64),(3,3),padding='same'))  #####Conv2D network

        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Dropout(hp.Choice('Dropout Prob',values=[0.05,0.08,0.1,0.13,0.16,0.20])))



    model.add(Flatten())



    for lays in range(hp.Int('Dense_layers_number',0,2,1)):                

        model.add(Dense(hp.Int('Dense_layer_size',64,256,64)))

        model.add(Activation('relu'))

        model.add(Dropout(hp.Choice('Dropout Prob',values=[0.05,0.08,0.1,0.13,0.16,0.20])))



    model.add(Dense(1))

    model.add(Activation('sigmoid'))



    ###########################################

    ###############  Model Compilation  #######



    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])

    return model 





        

tuner = kt.RandomSearch(bui_mod,

                     objective="val_accuracy",

                     max_trials=1,

                     executions_per_trial=1,

                     directory='keras_testing' 

                    )



print('1')

tuner.search(x=X,

             y=y,

             epochs=1,

             batch_size=64,

             validation_split=0.15)



print('2')

print(tuner.results_summary())



print('3')

models = tuner.get_best_models(num_models=1)[0]

print('4')



train_dir = "test1"

path = os.path.join(main_dir,train_dir)

#os.listdir(path)

print('5')

X_test = []

print('6')

id_line = []

def create_test1_data(path):

    for p in os.listdir(path):

        id_line.append(p.split(".")[0])

        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

        new_img_array = cv2.resize(img_array, dsize=(80, 80))

        X_test.append(new_img_array)

create_test1_data(path)

X_test = np.array(X_test).reshape(-1,80,80,1)

X_test = X_test/255







predictions = models.predict(X_test)



predicted_val = [int(round(p[0])) for p in predictions]



submission_df = pd.DataFrame({'id':id_line, 'label':predicted_val})



savepickle(submission_df,'submissionspickle')



submission_df.to_csv("./submission.csv", index=False)


