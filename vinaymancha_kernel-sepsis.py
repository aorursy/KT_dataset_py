import os

import pandas as pd

import numpy as np

import math

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM
path = '/kaggle/input/physionet-challenge-2019early-detection-of-sepsis'

path_a = '/kaggle/input/physionet-challenge-2019early-detection-of-sepsis/training_setA/training'

path_b = '/kaggle/input/physionet-challenge-2019early-detection-of-sepsis/training_setB/training_setB'

y_train = []

X_train = []



for i in os.listdir(path_a):

    data = pd.read_csv(path_a+'/'+i,sep = '|')

    data.drop(['EtCO2','Fibrinogen', 'Unit1', 'Unit2', 'BaseExcess', 'DBP', 'Hct', 'Hgb', 'PTT', 'WBC', 'pH','HCO3','FiO2', 'PaCO2', 'Platelets', 'Magnesium',  'Phosphate',  'Potassium', 'Bilirubin_total',  'TroponinI','SaO2', 'AST','BUN', 'Alkalinephos', 'Bilirubin_direct','Glucose','Lactate', 'Calcium',  'Chloride', 'Creatinine' ],axis = 1,inplace = True)



    data.dropna(thresh=data.shape[1]*0.40,how='all',inplace = True)

    La_1 = data['SepsisLabel'].sum()

    if La_1:

        y_train.append(1)

    else:

        y_train.append(0)

    data.drop(['SepsisLabel'],axis = 1,inplace = True)

    data = data.apply(lambda x: x.fillna(x.median()),axis=0)

    data = data.fillna(0)

    if len(data) < 40:

        Pad = pd.DataFrame({'HR':0.0 ,'O2Sat':0.0, 'Temp':0.0 , 'SBP':0.0, 'MAP':0.0, 'Resp':0.0, 'Age':0.0, 'Gender': 0 ,'HospAdmTime':0.0, 'ICULOS':0}, index =[item for item in range(0,40-len(data))])

        data = pd.concat([Pad, data]).reset_index(drop = True)

    elif len(data) >40:

        data = data[len(data)-40::1]

    data = data.values

    X_train.append(data)
for i in os.listdir(path_b):

    data = pd.read_csv(path_b+'/'+i,sep = '|')

    data.drop(['EtCO2','Fibrinogen', 'Unit1', 'Unit2', 'BaseExcess', 'DBP', 'Hct', 'Hgb', 'PTT', 'WBC', 'pH','HCO3','FiO2', 'PaCO2', 'Platelets', 'Magnesium',  'Phosphate',  'Potassium', 'Bilirubin_total',  'TroponinI','SaO2', 'AST','BUN', 'Alkalinephos', 'Bilirubin_direct','Glucose','Lactate', 'Calcium',  'Chloride', 'Creatinine' ],axis = 1,inplace = True)



    data.dropna(thresh=data.shape[1]*0.40,how='all',inplace = True)

    La_1 = data['SepsisLabel'].sum()

    if La_1:

        y_train.append(1)

    else:

        y_train.append(0)

    data.drop(['SepsisLabel'],axis = 1,inplace = True)

    data = data.apply(lambda x: x.fillna(x.median()),axis=0)

    data = data.fillna(0)

    if len(data) < 40:

        Pad = pd.DataFrame({'HR':0.0 ,'O2Sat':0.0, 'Temp':0.0 , 'SBP':0.0, 'MAP':0.0, 'Resp':0.0, 'Age':0.0, 'Gender': 0 ,'HospAdmTime':0.0, 'ICULOS':0}, index =[item for item in range(0,40-len(data))])

        data = pd.concat([Pad, data]).reset_index(drop = True)

    elif len(data) >40:

        data = data[len(data)-40::1]

    data = data.values

    X_train.append(data)
X, y = np.array(X_train) , np.array(y_train)
print(len(X))
from sklearn.model_selection import train_test_split





X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.25, random_state=45)

print(len(X_train_))
print(len(X_train_))

print(len(X_test_))
model = Sequential()

model.add(LSTM(128, input_shape=(40,10), return_sequences = True))

model.add(LSTM(128))

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_, y_train_, epochs=3, batch_size=32, verbose=1, validation_split=0.2, shuffle=True)
import matplotlib.pyplot as plt





# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")
'''

from keras.models import model_from_json

# load json and create model

json_file = open('/kaggle/working/model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("/kaggle/working/model.h5")

print("Loaded model from disk")'''
#y_pred = loaded_model.predict(X_test_)
y_pred = model.predict(X_test_)
from collections import Counter

from scipy import stats



dist = Counter(y)

for k in dist:

    dist[k] /= len(X)



acum = 0

bound = {}

for i in range(1):

    acum += dist[i]

    bound[i] = np.percentile(y_pred, acum * 100)

print(bound)



def classify(x):

    if x <= bound[0]:

        return 0

    else:

        return 1

    

final_pred = np.array(list(map(classify, y_pred)))

print(final_pred)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test_, final_pred)
uniqueValues, occurCount = np.unique(y_pred, return_counts=True)

occurCount
uniqueValues, occurCount = np.unique(y_test_, return_counts=True)

occurCount