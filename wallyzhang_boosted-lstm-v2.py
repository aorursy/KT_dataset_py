import numpy as np 

import pandas as pd 

from sklearn import model_selection

from sklearn.metrics import accuracy_score

from collections import Counter

import keras

from keras.models import Sequential 

from keras.layers import Activation, MaxPooling1D, Dropout, Flatten, Reshape, Dense, Conv1D, LSTM,SpatialDropout1D

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

import numpy as np

import pandas as pd

import seaborn as sns

import os

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, f1_score

from sklearn.model_selection import train_test_split

from random import randrange

from random import seed

from random import random

import pickle

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
files = dictionary = {'ADLOAD' : 162, 'AGENT' : 184, 'ALLAPLE_A' : 986, 'BHO' : 332, 'BIFROSE' : 156, 'CEEINJECT' : 873, 'CYCBOT_G' : 597, 'FAKEREAN' : 553,

                  'HOTBAR' : 129, 'INJECTOR' : 158, 'ONLINEGAMES' : 210, 'RENOS' : 532, 'RIMECUD_A' : 153, 'SMALL' : 180, 

                  'TOGA_RFN' : 406, 'VB' : 346, 'VBINJECT' : 937, 'VOBFUS' : 929 , 'VUNDO' : 762, 'WINWEBSEC' : 837, 'ZBOT' : 303}



dataSelect = {'ADLOAD' : (0,130), 'AGENT' : (161, 310), 'ALLAPLE_A' : (345, 1036), 'BHO' : (1331, 1582), 'BIFROSE' : (1663, 1790), 'CEEINJECT' : (1819, 2430), 'CYCBOT_G' : (2692, 3140), 'FAKEREAN' : (3290, 3705),

              'HOTBAR' : (3842, 3945), 'INJECTOR' : (3972, 4098), 'ONLINEGAMES' : (4129, 4297), 'RENOS' : (4339, 4738), 'RIMECUD_A' : (4871, 4993), 'SMALL' : (5024, 5168), 

              'TOGA_RFN' : (5204, 5508), 'VB' : (5610, 5869), 'VBINJECT' : (5956, 6612), 'VOBFUS' : (6893, 7543) , 'VUNDO' : (7822, 8355), 'WINWEBSEC' : (8584, 9170), 'ZBOT' : (9421, 9648)}



dataSelect2 = {'ADLOAD' : (0,161), 'AGENT' : (161, 345), 'ALLAPLE_A' : (345, 1331), 'BHO' : (1331, 1663), 'BIFROSE' : (1663, 1819), 'CEEINJECT' : (1819, 2692), 'CYCBOT_G' : (2692, 3290), 'FAKEREAN' : (3290, 3842),

              'HOTBAR' : (3842, 3972), 'INJECTOR' : (3972, 4129), 'ONLINEGAMES' : (4129, 4339), 'RENOS' : (4339, 4871), 'RIMECUD_A' : (4871, 5024), 'SMALL' : (5024, 5204), 

              'TOGA_RFN' : (5204, 5610), 'VB' : (5610, 5956), 'VBINJECT' : (5956, 6893), 'VOBFUS' : (6893, 7822) , 'VUNDO' : (7822, 8584), 'WINWEBSEC' : (8584, 9421), 'ZBOT' : (9421, 9724)}



families = [ 'ADLOAD', 'AGENT' , 'ALLAPLE_A', 'BHO', 'BIFROSE', 'CEEINJECT', 'CYCBOT_G','FAKEREAN', 'HOTBAR', 'INJECTOR',

            'ONLINEGAMES', 'RENOS', 'RIMECUD_A', 'SMALL', 'TOGA_RFN', 'VB', 'VBINJECT',

            'VOBFUS', 'VUNDO', 'WINWEBSEC', 'ZBOT']
df = pd.read_csv('/kaggle/input/final-opcodes/all_data.csv')

print(df.shape)

df
def subsample(X, Y, errors, ratio=1.0):

    sampleX = np.empty((0,0), dtype = np.int8)

    sampleY = np.empty((0,0), dtype = np.int8)

    for i in errors:

        sampleX = np.append(sampleX, X.iloc[i, :].values)

        sampleY = np.append(sampleY, Y.iloc[i])

    

    n_sample = round(len(Y) * ratio)

    while len(sampleY) < n_sample:

        index = randrange(Y.shape[0])

        X_row = X.iloc[index, :].values

        Y_row = Y[index]

        sampleX = np.append(sampleX, X_row)

        sampleY = np.append(sampleY, Y_row)

    arr = [sampleX, sampleY]

    return arr
def plot_acc(h):

    plt.plot(h.history['accuracy'])

    plt.plot(h.history['val_accuracy'])



    plt.title('model accuracy')

    plt.ylabel('accuracy and loss')

    plt.xlabel('epoch')



    plt.legend(['acc', 'val acc' ], loc='upper left')

    plt.show()
def plot_loss(h):

    plt.plot(h.history['loss'])

    plt.plot(h.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('accuracy and loss')

    plt.xlabel('epoch')



    plt.legend(['loss', 'val loss' ], loc='upper left')

    plt.show()
model = Sequential()

model.add(LSTM(512, dropout=0,  recurrent_dropout=0,go_backwards=True, input_shape=(1000,1)))

model.add(Dropout(0.2))

model.add(Dense(21,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Getting X and Y Data

X = df.iloc[:, 34:]

Y = df.iloc[:, 1]



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Y = le.fit_transform(Y)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 23)



print(X_test.shape)

X_test = tf.reshape(X_test, (X_test.shape[0], 1000, 1))
acc = []

valacc = []



#uncommentted

file = '/kaggle/input/boost-all/errors3.sav' #increment

errors = pickle.load(open(file, 'rb'))



print("training model", 0, "--------------------------")

sample = subsample(X_train, y_train, [], ratio=0.6)

baggingSampleX = sample[0].reshape(-1, 1000)

baggingSampleY = sample[1]



#print(X_train.shape)

#print(X_test.shape)

print(baggingSampleX.shape)



baggingSampleX = tf.reshape(baggingSampleX, (baggingSampleX.shape[0], 1000, 1))



#print(baggingSampleX.shape)

#print(X_test.shape)

history = model.fit(baggingSampleX,baggingSampleY, epochs = 21,batch_size=1,validation_data = (X_test, y_test), shuffle = True)





accuracy_logs = history.history["accuracy"]

val_accuracy_logs = history.history["val_accuracy"]

acc.append(accuracy_logs)

valacc.append(accuracy_logs)



file_name = "boosted_lstm_4"  #increment



json = file_name + ".json"

h5 = file_name + ".h5"



model_json = model.to_json()

with open(json, "w") as json_file:

    json_file.write(model_json)

    model.save_weights(h5)
def checkPred(array):

    bestScore = -1

    count = -1

    best_model = -1

    for i in range(21):

        if array[i] > count:

            count = array[i]

            best_model = i

#             bestScore = scores[i]

#         elif array[i] == count and scores[i] > bestScore: --> USE IF CAN OBTAIN CONFIDENCE LEVEL

#             best_model = i

#             bestScore = scores[i]

    return best_model
#Getting X and Y Data

X = df.iloc[:, 34:]

Y = df.iloc[:, 1]



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Y = le.fit_transform(Y)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 23)



print(X_test.shape)

X_train = tf.reshape(X_train, (X_train.shape[0], 1000, 1))

X_test = tf.reshape(X_test, (X_test.shape[0], 1000, 1))

print(X_train.shape)

print(X_test.shape)





preds = model.predict_classes(X_train) #current model

print(preds.shape)



predFile = "/kaggle/working/y_pred" + str(4) + ".sav"

pickle.dump(preds, open(predFile, 'wb'))

Y_pred = np.empty(0, dtype=np.int8)

for i in range(7780):

    predFile = "/kaggle/input/boost-all/y_pred"

    array = [0] * 21

    for j in range (0,4): #increment

        file = predFile + str(j) + ".sav"

        y_pred = pickle.load(open(file, 'rb'))

        array[y_pred[i]] += 1

    predFile = "/kaggle/working/y_pred4.sav"

    y_pred = pickle.load(open(predFile, 'rb'))

    array[y_pred[i]] += 1

    final_Pred = checkPred(array)

    Y_pred = np.append(Y_pred, final_Pred)

    print(families[final_Pred])

    array = [0] * 21
print(Y_pred.shape)

predBoostFile = "/kaggle/working/y_combined_pred_4.sav" #increment

pickle.dump(Y_pred, open(predBoostFile, 'wb'))
from sklearn.metrics import accuracy_score

print("-------------------------")



print(accuracy_score(y_train, Y_pred))





error = [np.empty((0,0), dtype = np.int8)] * 21

for i in range(7284):

    if(y_train[i] != Y_pred[i]):

        error = np.append(error, i)
errorFile = "/kaggle/working/errors4.sav" #increment

pickle.dump(error, open(errorFile, 'wb'))