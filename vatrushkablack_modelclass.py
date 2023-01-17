import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

import tensorflow as tf

import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras

import pandas_profiling as pp







sc = MinMaxScaler(feature_range = (0, 1))



rock_dataset = pd.read_csv("../input/0.csv", header=None) # class = 0

scissors_dataset = pd.read_csv("../input/1.csv", header=None) # class = 1

paper_dataset = pd.read_csv("../input/2.csv", header=None) # class = 2

ok_dataset = pd.read_csv("../input/3.csv", header=None) # class = 3



print("Rock Shape: ",rock_dataset.shape,

      "\nScissor Shape: ",scissors_dataset.shape,

      "\nPaper Shape: ",paper_dataset.shape,

      "\nOK Shape: ",ok_dataset.shape)





rock_dataset.head(3)
scissors_dataset.head(3)
paper_dataset.head(3)
ok_dataset.head(3)
colors=["forestgreen","teal","crimson","chocolate","darkred","lightseagreen","orangered","chartreuse"]

time_rock=rock_dataset.iloc[:,0:8]

time_rock.index=pd.to_datetime(time_rock.index)

time_rock.iloc[:170,:].plot(subplots=True,figsize=(10,10),colors=colors);
time_scis=scissors_dataset.iloc[:,0:8]

time_scis.index=pd.to_datetime(time_scis.index)

time_scis.iloc[:170,:].plot(subplots=True,figsize=(10,10),colors=colors);
time_paper=paper_dataset.iloc[:,0:8]

time_paper.index=pd.to_datetime(time_paper.index)

time_paper.iloc[:170,:].plot(subplots=True,figsize=(10,10),colors=colors);
time_ok=ok_dataset.iloc[:,0:8]

time_ok.index=pd.to_datetime(time_ok.index)

time_ok.iloc[:170,:].plot(subplots=True,figsize=(10,10),colors=colors);
frames = [rock_dataset, scissors_dataset, paper_dataset, ok_dataset]

dataset = pd.concat(frames)



dataset_train = dataset.iloc[np.random.permutation(len(dataset))]

dataset_train.reset_index(drop=True)



X_train = []

y_train = []



for i in range(0, dataset_train.shape[0]):

    row = np.array(dataset_train.iloc[i:1+i, 0:64].values)

    X_train.append(np.reshape(row, (64, 1)))

    y_train.append(np.array(dataset_train.iloc[i:1+i, -1:])[0][0])



X_train = np.array(X_train)

y_train = np.array(y_train)



# Reshape to one flatten vector

X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1], 1)

X_train = sc.fit_transform(X_train)



# Reshape again after normalization to (-1, 8, 8)

X_train = X_train.reshape((-1, 8, 8))



# Convert to one hot

y_train = np.eye(np.max(y_train) + 1)[y_train]





print("All Data size X and y")

print(X_train.shape)

print(y_train.shape)

# Splitting Train/Test

X_test = X_train[7700:]

y_test = y_train[7700:]

print("Test Data size X and y")

print(X_test.shape)

print(y_test.shape)



X_train = X_train[0:7700]

y_train = y_train[0:7700]

print("Train Data size X and y")

print(X_train.shape)

print(y_train.shape)



# Creating the model



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dropout



classifier = Sequential()



classifier.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 8)))

classifier.add(Dropout(0.2))



classifier.add(LSTM(units = 50, return_sequences = True))

classifier.add(Dropout(0.2))



classifier.add(LSTM(units = 50, return_sequences = True))

classifier.add(Dropout(0.2))



classifier.add(LSTM(units = 50))

classifier.add(Dropout(0.2))



classifier.add(Dense(units = 64))

classifier.add(Dense(units = 128))



classifier.add(Dense(units = 4, activation="softmax"))



classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy')



classifier.fit(X_train, y_train, epochs = 250, batch_size = 32, verbose=2)



# Save

classifier.save("model_cross_splited_data.h5")

print("Saved model to disk")



###############################################



from tensorflow import keras



# # Load Model

# model = keras.models.load_model('model_cross_splited_data.h5')

# model.summary()



def evaluateModel(prediction, y):

    good = 0

    for i in range(len(y)):

        if (prediction[i] == np.argmax(y[i])):

            good = good +1

    return (good/len(y)) * 100.0



result_test = classifier.predict_classes(X_test)

print("Correct classification rate on test data")

print(evaluateModel(result_test, y_test))



result_train = classifier.predict_classes(X_train)

print("Correct classification rate on train data")

print(evaluateModel(result_train, y_train))

col_names = list()

for i in range(0,65):

    if i == 64:

        col_names.append("class")

    else:

        col_names.append("sensor"+str(i+1))

        

rock_dataset.columns = col_names

scissors_dataset.columns = col_names

paper_dataset.columns = col_names

ok_dataset.columns = col_names



dataset_2 = pd.concat([rock_dataset,scissors_dataset,paper_dataset,ok_dataset],ignore_index=True)

print(dataset_2.tail())

print(dataset_2.shape)
from sklearn.neighbors import KNeighborsClassifier



y = dataset_2["class"].values.reshape(-1,1)

x = dataset_2.drop(["class"],axis = 1).values



total = dataset_2.isnull().sum().sort_values(ascending = False)

percentage = (dataset_2.isnull().sum()/dataset_2.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total,percentage],axis = 1, keys = ["Total","Percentage"])

missing_data
dataset_2.describe()
#from pandas_profiling import 

#ProfileReportprofile = ProfileReport(dataset_2, title='Pandas Profiling Report')

#pp.ProfileReport(dataset_2)
sns.boxplot(x = dataset_2['sensor32'])
#next step is to standardize our data - using MinMaxScaler

y = dataset_2["class"]

x_data1 = dataset_2.drop(["class"],axis = 1)



from sklearn import preprocessing



scaler = preprocessing.MinMaxScaler()

scaler.fit(x_data1)



x_data1 = pd.DataFrame(scaler.transform(x_data1), index=x_data1.index, columns=x_data1.columns)

pp.ProfileReport(x_data1)
y = dataset_2["class"].values.reshape(-1,1)

x = dataset_2.drop(["class"],axis = 1).values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 42)



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)  #n_neighbor = k

knn.fit(x_train,y_train)



#Find the best K value



k_value = []

accuracy = []



for i in range(1,22):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,np.ravel(y_train,order='C'))

    

    score = knn.score(x_test,y_test)

    k_value.append(i)

    accuracy.append(score)

for i,j in zip(k_value,accuracy):

    print(i,j)



#Find K value for Max accuracy

plt.plot(range(1,22),accuracy,color = "blue")

plt.xlabel("K value")

plt.ylabel("Accuracy")

plt.show()