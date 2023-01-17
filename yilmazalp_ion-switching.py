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
data = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

eval_data = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
data.head()
eval_data.head()
subsets = []

eval_subsets = []



part_number = int(data.shape[0]/500000)

eval_part_number = int(eval_data.shape[0]/500000)



for part in range(1, part_number+1):

    subsets.append(part)

    subsets[part-1] = data.iloc[((part-1)*500000):part*500000,:]

    

for eval_part in range(1, eval_part_number+1):

    eval_subsets.append(eval_part)

    eval_subsets[eval_part-1] = eval_data.iloc[((eval_part-1)*500000):eval_part*500000,:]
from matplotlib import pyplot as plt



plt.figure(figsize=(20,5)); res = 1000

plt.plot(range(0,data.shape[0],res),data.signal[0::res])

#plt.plot(range(0,data.shape[0],res),data.open_channels[0::res])



for i in range(11): plt.plot([i*500000,i*500000],[-5,12.5],'r')

for j in range(10): plt.text(j*500000+200000,10,str(j+1),size=20)

plt.xlabel('Row',size=16); plt.ylabel('Signal',size=16); 

plt.title('Training Data Signal - 10 batches',size=20)

plt.show()
#Crete a new clean data as copy of the data



clean_data = data.copy()



#Clean drift data in the batch 2 



a = 500000; b = 600000

clean_data.loc[data.index[a:b],'signal'] = clean_data.signal[a:b].values - 3*(clean_data.time.values[a:b] - 50)/10.
def sin(x, A, phi, delta):

    frequency = 0.01

    omega = 2 * np.pi * frequency

    return A * np.sin(omega * x + phi) + delta
import math

def parabolic_drift_fit(data):

    x = data['time']

    y = data['signal']



    frequency = 0.01

    omega = 2 * np.pi * frequency

    

    #construct matrix M for every data point t in the time data x

    M = np.array([[np.sin(omega * time_point), np.cos(omega * time_point), 1] for time_point in x])

    

    #construct the signal data y

    y = np.array(y).reshape(len(y), 1)



    #linalg.lstsq function solves the equation a x = b by computing a vector x 

    #that minimizes the squared Euclidean 2-norm

    (theta, _, _, _) = np.linalg.lstsq(M, y)

    

    #find A, phi, and delta parameters

    A = np.sqrt(theta[0,0]**2 + theta[1,0]**2)

    phi = math.atan2(theta[1,0], theta[0,0])

    delta = theta[2,0]



    optimal_parabol_drift = [A, phi, delta]

    

    

    return optimal_parabol_drift
#find optimal parameter values for batch 7

a = 500000*6; b = 500000*7

print(parabolic_drift_fit(data.iloc[a:b]))



#find optimal parameter values for batch 8

a = 500000*7; b = 500000*8

print(parabolic_drift_fit(data.iloc[a:b]))



#find optimal parameter values for batch 9

a = 500000*8; b = 500000*9

print(parabolic_drift_fit(data.iloc[a:b]))



#find optimal parameter values for batch 10

a = 500000*9; b = 500000*10

print(parabolic_drift_fit(data.iloc[a:b]))
def remove_parabolic_drift(x, A, opt_phi):

    frequency = 0.01

    omega = 2 * np.pi * frequency

    y = A * np.sin(omega * x + opt_phi)

    return y
#Clean drift data in the batch 7

a = 500000*6; b = 500000*7

clean_data.loc[data.index[a:b],'signal'] = data.signal.values[a:b] - remove_parabolic_drift(data.time[a:b].values, 

                                                                                            4.99, 0)



#Clean drift data in the batch 8

a = 500000*7; b = 500000*8

clean_data.loc[data.index[a:b],'signal'] = data.signal.values[a:b] - remove_parabolic_drift(data.time[a:b].values, 

                                                                                            5.07, 3.138)



#Clean drift data in the batch 9

a = 500000*8; b = 500000*9

clean_data.loc[data.index[a:b],'signal'] = data.signal.values[a:b] - remove_parabolic_drift(data.time[a:b].values, 

                                                                                            4.96, 0)



#Clean drift data in the batch 10

a = 500000*9; b = 500000*10

clean_data.loc[data.index[a:b],'signal'] = data.signal.values[a:b] - remove_parabolic_drift(data.time[a:b].values, 

                                                                                            5.07, 3.136)



plt.figure(figsize=(20,5))

plt.plot(data.time[::1000],data.signal[::1000])

plt.title('Training Batches with Drift',size=16)





plt.figure(figsize=(20,5))

plt.plot(clean_data.time[::1000],clean_data.signal[::1000])

plt.title('Training Batches without Drift',size=16)

plt.show()
from matplotlib import pyplot as plt



plt.figure(figsize=(20,5)); res = 1000

#plt.plot(range(0,data.shape[0],res),data.signal[0::res])

plt.plot(range(0,data.shape[0],res),data.open_channels[0::res])



for i in range(11): plt.plot([i*500000,i*500000],[-5,12.5],'r')

for j in range(10): plt.text(j*500000+200000,10,str(j+1),size=20)

plt.xlabel('Row',size=16); plt.ylabel('Signal',size=16); 

plt.title('Training Data Signal - 10 batches',size=20)

plt.show()
from collections import Counter



part_number = int(data.shape[0]/500000)

ion_channels_number_max = []

ion_channels_number_min = []





for sub_index in range(part_number):

    sub_target_set = subsets[sub_index].iloc[:,-1]

    ion_channels_number_max.append(max(Counter(sub_target_set).keys()))

    ion_channels_number_min.append(min(Counter(sub_target_set).keys()))
ion_channels_number_max
ion_channels_number_min
from matplotlib import pyplot as plt



plt.figure(figsize=(20,5)); res = 1000

plt.plot(range(0,clean_data.shape[0],res),clean_data.signal[0::res])

plt.plot(range(0,data.shape[0],res),data.open_channels[0::res])



for i in range(11): plt.plot([i*500000,i*500000],[-5,12.5],'r')

for j in range(10): plt.text(j*500000+200000,10,str(j+1),size=20)

plt.xlabel('Row',size=16); plt.ylabel('Signal',size=16); 

plt.title('Training Data Signal - 10 batches',size=20)

plt.show()
clean_subsets = []

#eval_subsets = []



part_number = int(clean_data.shape[0]/500000)

#eval_part_number = int(eval_data.shape[0]/500000)



for part in range(1, part_number+1):

    clean_subsets.append(part)

    clean_subsets[part-1] = clean_data.iloc[((part-1)*500000):part*500000,:]
print("variance of batch 1 : ", clean_subsets[0].signal.var())

print("variance of batch 2 : ", clean_subsets[1].signal.var())

print("\n")

print("variance of batch 3 : ", clean_subsets[2].signal.var())

print("variance of batch 7 : ", clean_subsets[6].signal.var())

print("\n")

print("variance of batch 4 : ", clean_subsets[3].signal.var())

print("variance of batch 8 : ", clean_subsets[7].signal.var())

print("\n")

print("variance of batch 5 : ", clean_subsets[4].signal.var())

print("variance of batch 10 : ", clean_subsets[9].signal.var())

print("\n")

print("variance of batch 6 : ", clean_subsets[5].signal.var())

print("variance of batch 9 : ", clean_subsets[8].signal.var())

print("\n")
from collections import Counter

from keras.utils import to_categorical

from keras.layers import LSTM, Dense, Dropout, Flatten, BatchNormalization, Conv1D, Activation, RepeatVector, TimeDistributed

from keras.models import Sequential, Input, Model

from keras.regularizers import l1, l2, l1_l2





class lstm_model:

    

    def __init__(self, part1, part2):

        self.part1 = part1

        self.part2 = part2

        #self.eval_subset_data = eval_subset_data

        

    def create_features(self):

        part1 = self.part1

        part2 = self.part2

        

        features = np.concatenate([clean_subsets[part1].signal, clean_subsets[part2].signal])

        features = np.expand_dims(features, axis = 1)

        

        return features

        

    def create_target(self):

        part1 = self.part1

        part2 = self.part2

        #target_set = self.subset_data.iloc[:,-1]

        

        target = np.concatenate([clean_subsets[part1].open_channels, clean_subsets[part2].open_channels])



        

        #target = np.array(target_set)

        #target = to_categorical(target)

        

        return target

    

    def train_size(self, rate):

    

        all_size = 1000000

        train_end = int(all_size*rate)

        

        return train_end

        

    

    def create_train_features(self, train_rate):

        features = self.create_features()

        train_end = self.train_size(train_rate)

        start = 0

        

        train_features = features[start:train_end]

        train_features = np.expand_dims(train_features, axis = 1)

        

        return train_features

    

    

    def create_test_features(self, train_rate):

        features = self.create_features()

        train_end = self.train_size(train_rate)

        end = features.shape[0]

        

        test_features = features[train_end:end]

        test_features = np.expand_dims(test_features, axis = 1)

        

        return test_features

    

    def create_train_target(self, train_rate):

        target = self.create_target()

        train_end = self.train_size(train_rate)

        start = 0

        

        train_target = target[start:train_end]

        train_target = to_categorical(train_target)

        

        return train_target

    

    def create_test_target(self, train_rate):

        features = self.create_features()

        target = self.create_target()

        train_end = self.train_size(train_rate)

        end = features.shape[0]

        

        test_target = target[train_end:end]

        test_target = to_categorical(test_target)

        

        return test_target

    

    

    def construct_model_for_batch_1_2(self):

        features = self.create_features()

        index = self.part1

        

        maximum_channel_number = ion_channels_number_max[index]

        

        model  = Sequential()



        #model.add(Conv1D(1, 1, input_shape=(features.shape[1], 1), activation='relu'))

        model.add(LSTM(1, input_shape=(features.shape[1], 1), return_sequences=False, activation='tanh'))

        model.add(Dense(maximum_channel_number+1, activation='sigmoid'))

        

        return model

    

    def construct_model_for_batch_3_7(self):

        features = self.create_features()

        index = self.part1

        

        maximum_channel_number = ion_channels_number_max[index]

        

        model  = Sequential()

        

        #model.add(Conv1D(4, 1, input_shape=(features.shape[1], 1), activation='relu'))

        model.add(LSTM(8, input_shape=(features.shape[1], 1), return_sequences=False, activation='tanh'))

        model.add(Dense(maximum_channel_number+1, activation='sigmoid'))

        

        return model

    

    def construct_model_for_batch_4_8(self):

        features = self.create_features()

        index = self.part1

        

        maximum_channel_number = ion_channels_number_max[index]

        

        model  = Sequential()



        model.add(Conv1D(16, 1, input_shape=(features.shape[1], 1), activation='relu'))

        model.add(LSTM(16, return_sequences=False, activation='tanh'))

        model.add(Dense(maximum_channel_number+1, activation='sigmoid'))

        

        return model

    

    def construct_model_for_batch_6_9(self):

        features = self.create_features()

        index = self.part1

        

        maximum_channel_number = ion_channels_number_max[index]

        

        model  = Sequential()



        model.add(Conv1D(24, 1, input_shape=(features.shape[1], 1), activation='relu'))

        model.add(LSTM(24, return_sequences=False, activation='tanh'))

        model.add(Dense(maximum_channel_number+1, activation='sigmoid'))

        

        return model

    

    def construct_model_for_batch_5_10(self):

        features = self.create_features()

        index = self.part1

        

        maximum_channel_number = ion_channels_number_max[index]

        

        #model  = Sequential()



        #model.add(Conv1D(128, 1, input_shape=(features.shape[1], 1), activation='relu'))

        

        input_data = Input(shape = (features.shape[1], 1))

        encoded = LSTM(12, activation='tanh', return_sequences=True)(input_data)

        encoded = LSTM(24, activation='tanh', return_sequences=True)(encoded)

        encoded = LSTM(48, activation='tanh', return_sequences=False)(encoded)

        

        decoded = RepeatVector(features.shape[1])(encoded)

        

        decoded = LSTM(48, activation='tanh', return_sequences=True)(decoded)

        decoded = LSTM(24, activation='tanh', return_sequences=True)(decoded)

        decoded = LSTM(12, activation='tanh', return_sequences=True)(decoded)



        output_data = LSTM(maximum_channel_number+1, activation = 'sigmoid', return_sequences=False)(decoded)

        

        model = Model(input_data, output_data)



        

        #model.add(LSTM(24, input_shape=(features.shape[1], 1), return_sequences=True, activation='tanh'))

        #model.add(LSTM(12, return_sequences=False, activation='tanh'))

        #model.add(RepeatVector(features.shape[1]))

        #model.add(LSTM(maximum_channel_number+1, return_sequences = True, activation='tanh'))

        

        #model.add(LSTM(24, return_sequences=False, activation='tanh'))



        #model.add(Dense(maximum_channel_number+1, activation='sigmoid'))

        

        return model
train_split_rate = 0.85

model1 = lstm_model(part1=0, part2=1)

processed_model1 = model1.construct_model_for_batch_1_2()



train_feature_model1 = model1.create_train_features(train_split_rate)

train_target_model1 = model1.create_train_target(train_split_rate)



test_feature_model1 = model1.create_test_features(train_split_rate)

test_target_model1 = model1.create_test_target(train_split_rate)





processed_model1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



rnn = processed_model1.fit(train_feature_model1, train_target_model1, batch_size=1000, epochs=20, 

                           validation_data=(test_feature_model1, test_target_model1))
#Plot the Loss Curves

plt.figure(figsize=[8,6])

train_loss, = plt.plot(rnn.history['loss'],'r',linewidth=3.0)

test_loss, = plt.plot(rnn.history['val_loss'],'b',linewidth=3.0)

plt.legend([train_loss, test_loss], ['Training Loss', 'Test Loss'],fontsize=12)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves',fontsize=16)

 

#Plot the Accuracy Curves

plt.figure(figsize=[8,6])

train_accuracy, = plt.plot(rnn.history['accuracy'],'r',linewidth=3.0)

test_accuracy, = plt.plot(rnn.history['val_accuracy'],'b',linewidth=3.0)

plt.legend([train_accuracy, test_accuracy], ['Training Accuracy', 'Test Accuracy'],fontsize=12)



plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Accuracy',fontsize=16)

plt.title('Accuracy Curves',fontsize=16)
#construct the lstm model for batch 3 and 7



train_split_rate = 0.67

model2 = lstm_model(2, 6)

processed_model2 = model2.construct_model_for_batch_3_7()



train_feature_model2 = model2.create_train_features(train_split_rate)

train_target_model2 = model2.create_train_target(train_split_rate)



test_feature_model2 = model2.create_test_features(train_split_rate)

test_target_model2 = model2.create_test_target(train_split_rate)





processed_model2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



rnn = processed_model2.fit(train_feature_model2, train_target_model2, batch_size=1000, epochs=25, 

                           validation_data=(test_feature_model2, test_target_model2))
#Plot the Loss Curves

plt.figure(figsize=[8,6])

train_loss, = plt.plot(rnn.history['loss'],'r',linewidth=3.0)

test_loss, = plt.plot(rnn.history['val_loss'],'b',linewidth=3.0)

plt.legend([train_loss, test_loss], ['Training Loss', 'Test Loss'],fontsize=12)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves',fontsize=16)

 

#Plot the Accuracy Curves

plt.figure(figsize=[8,6])

train_accuracy, = plt.plot(rnn.history['accuracy'],'r',linewidth=3.0)

test_accuracy, = plt.plot(rnn.history['val_accuracy'],'b',linewidth=3.0)

plt.legend([train_accuracy, test_accuracy], ['Training Accuracy', 'Test Accuracy'],fontsize=12)



plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Accuracy',fontsize=16)

plt.title('Accuracy Curves',fontsize=16)
#construct the lstm model for batch 4 and 8



train_split_rate = 0.75

model3 = lstm_model(3, 7)

processed_model3 = model3.construct_model_for_batch_4_8()



train_feature_model3 = model3.create_train_features(train_split_rate)

train_target_model3 = model3.create_train_target(train_split_rate)



test_feature_model3 = model3.create_test_features(train_split_rate)

test_target_model3 = model3.create_test_target(train_split_rate)





#opt = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

processed_model3.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])





rnn = processed_model3.fit(train_feature_model3, train_target_model3, batch_size=1000, epochs=40, 

                           validation_data=(test_feature_model3, test_target_model3))
#Plot the Loss Curves

plt.figure(figsize=[8,6])

train_loss, = plt.plot(rnn.history['loss'],'r',linewidth=3.0)

test_loss, = plt.plot(rnn.history['val_loss'],'b',linewidth=3.0)

plt.legend([train_loss, test_loss], ['Training Loss', 'Test Loss'],fontsize=12)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves',fontsize=16)

 

#Plot the Accuracy Curves

plt.figure(figsize=[8,6])

train_accuracy, = plt.plot(rnn.history['accuracy'],'r',linewidth=3.0)

test_accuracy, = plt.plot(rnn.history['val_accuracy'],'b',linewidth=3.0)

plt.legend([train_accuracy, test_accuracy], ['Training Accuracy', 'Test Accuracy'],fontsize=12)



plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Accuracy',fontsize=16)

plt.title('Accuracy Curves',fontsize=16)
#construct the lstm model for batch 6 and 9



train_split_rate = 0.67

model4 = lstm_model(5, 8)

processed_model4 = model4.construct_model_for_batch_6_9()



train_feature_model4 = model4.create_train_features(train_split_rate)

train_target_model4 = model4.create_train_target(train_split_rate)



test_feature_model4 = model4.create_test_features(train_split_rate)

test_target_model4 = model4.create_test_target(train_split_rate)





processed_model4.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



rnn = processed_model4.fit(train_feature_model4, train_target_model4, batch_size=1000, epochs=25, 

                           validation_data=(test_feature_model4, test_target_model4))
#Plot the Loss Curves

plt.figure(figsize=[8,6])

train_loss, = plt.plot(rnn.history['loss'],'r',linewidth=3.0)

test_loss, = plt.plot(rnn.history['val_loss'],'b',linewidth=3.0)

plt.legend([train_loss, test_loss], ['Training Loss', 'Test Loss'],fontsize=12)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves',fontsize=16)

 

#Plot the Accuracy Curves

plt.figure(figsize=[8,6])

train_accuracy, = plt.plot(rnn.history['accuracy'],'r',linewidth=3.0)

test_accuracy, = plt.plot(rnn.history['val_accuracy'],'b',linewidth=3.0)

plt.legend([train_accuracy, test_accuracy], ['Training Accuracy', 'Test Accuracy'],fontsize=12)



plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Accuracy',fontsize=16)

plt.title('Accuracy Curves',fontsize=16)
model5 = lstm_model(4, 9)

processed_model5 = model5.construct_model_for_batch_5_10()
processed_model5.summary()
#construct the lstm model for batch 5 and 10

import keras

import numpy as np



train_split_rate = 0.75

model5 = lstm_model(4, 9)

processed_model5 = model5.construct_model_for_batch_5_10()



train_feature_model5 = model5.create_train_features(train_split_rate)

train_target_model5 = model5.create_train_target(train_split_rate)



test_feature_model5 = model5.create_test_features(train_split_rate)

test_target_model5 = model5.create_test_target(train_split_rate)





#opt = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

processed_model5.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



rnn = processed_model5.fit(train_feature_model5, train_target_model5, batch_size=1000, epochs=30,

                           validation_data=(test_feature_model5, test_target_model5))
#Plot the Loss Curves

plt.figure(figsize=[8,6])

train_loss, = plt.plot(rnn.history['loss'],'r',linewidth=3.0)

test_loss, = plt.plot(rnn.history['val_loss'],'b',linewidth=3.0)

plt.legend([train_loss, test_loss], ['Training Loss', 'Test Loss'],fontsize=12)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves',fontsize=16)

 

#Plot the Accuracy Curves

plt.figure(figsize=[8,6])

train_accuracy, = plt.plot(rnn.history['accuracy'],'r',linewidth=3.0)

test_accuracy, = plt.plot(rnn.history['val_accuracy'],'b',linewidth=3.0)

plt.legend([train_accuracy, test_accuracy], ['Training Accuracy', 'Test Accuracy'],fontsize=12)



plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Accuracy',fontsize=16)

plt.title('Accuracy Curves',fontsize=16)
plt.figure(figsize=(20,5))

let = ['A','B','C','D','E','F','G','H','I','J']



r = eval_data.signal.rolling(30000).mean()

plt.plot(eval_data.time.values,r)



for i in range(21): plt.plot([500+i*10,500+i*10],[-3,6],'r:')

for i in range(5): plt.plot([500+i*50,500+i*50],[-3,6],'r')

for k in range(4): plt.text(525+k*50,5.5,str(k+1),size=20)

for k in range(10): plt.text(505+k*10,4,let[k],size=16)

plt.title('Test Signal Rolling Mean. Has Drift wherever plot is not horizontal line',size=16)

plt.show()
# removing drift for the subparts of A, B, and E in the batch 1 



#removing drift in the A

start=500

a = 0; b = 100000

eval_data.loc[eval_data.index[a:b],'signal'] = eval_data.signal.values[a:b] - 3*(eval_data.time.values[a:b]-start)/10.



#removing drift in the B

start=510

a = 100000; b = 200000

eval_data.loc[eval_data.index[a:b],'signal'] = eval_data.signal.values[a:b] - 3*(eval_data.time.values[a:b]-start)/10.



#removing drift in the E

start=540

a = 400000; b = 500000

eval_data.loc[eval_data.index[a:b],'signal'] = eval_data.signal.values[a:b] - 3*(eval_data.time.values[a:b]-start)/10.
# removing drift for the subparts of G, H, and I in the batch 2 



#removing drift in the G

start=560

a = 600000; b = 700000

eval_data.loc[eval_data.index[a:b],'signal'] = eval_data.signal.values[a:b] - 3*(eval_data.time.values[a:b]-start)/10.



#removing drift in the H

start=570

a = 700000; b = 800000

eval_data.loc[eval_data.index[a:b],'signal'] = eval_data.signal.values[a:b] - 3*(eval_data.time.values[a:b]-start)/10.



#removing drift in the I

start=580

a = 800000; b = 900000

eval_data.loc[eval_data.index[a:b],'signal'] = eval_data.signal.values[a:b] - 3*(eval_data.time.values[a:b]-start)/10.
a = 1000000; b = 1500000

print(parabolic_drift_fit(eval_data.iloc[a:b]))
#removing drift for the data in the batch 3

a = 1000000; b = 1500000

eval_data.loc[eval_data.index[a:b],'signal'] = eval_data.signal.values[a:b] - remove_parabolic_drift(eval_data.time[a:b].values, 4.99, 0)


plt.figure(figsize=(20,5))

r = eval_data.signal.rolling(30000).mean()

plt.plot(eval_data.time.values,r)

for i in range(21): plt.plot([500+i*10,500+i*10],[-2,6],'r:')

for i in range(5): plt.plot([500+i*50,500+i*50],[-2,6],'r')

for k in range(4): plt.text(525+k*50,5.5,str(k+1),size=20)

for k in range(10): plt.text(505+k*10,4,let[k],size=16)

plt.title('Test Signal Rolling Mean without Drift',size=16)

plt.show()
plt.figure(figsize=(20,5))

res = 1000

plt.plot(range(0,eval_data.shape[0],res),eval_data.signal[0::res])

for i in range(5): plt.plot([i*500000,i*500000],[-5,12.5],'r')

for i in range(21): plt.plot([i*100000,i*100000],[-5,12.5],'r:')

for k in range(4): plt.text(k*500000+250000,10,str(k+1),size=20)

for k in range(10): plt.text(k*100000+40000,7.5,let[k],size=16)

plt.title('Test Signal without Drift',size=16)

plt.show()

plt.figure(figsize=(20,5))

plt.plot(clean_data.time[::1000],clean_data.signal[::1000])

plt.title('Training Batches 7-10 without Parabolic Drift',size=16)

plt.show()
submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
#prediction of the subparts of A,D,I, the parts of 3 and 4



from sklearn.metrics import f1_score, accuracy_score



eval_A = eval_data.signal.values[0:100000].reshape(-1,1)

eval_A = np.expand_dims(eval_A, axis = 2)



eval_D = eval_data.signal.values[300000:400000].reshape(-1,1)

eval_D = np.expand_dims(eval_D, axis = 2)



eval_I = eval_data.signal.values[800000:900000].reshape(-1,1)

eval_I = np.expand_dims(eval_I, axis = 2)



eval_3 = eval_data.signal.values[1000000:1500000].reshape(-1,1)

eval_3 = np.expand_dims(eval_3, axis = 2)



eval_4 = eval_data.signal.values[1500000:2000000].reshape(-1,1)

eval_4 = np.expand_dims(eval_4, axis = 2)





pred_A = processed_model1.predict_classes(eval_A)

pred_D = processed_model1.predict_classes(eval_D)

pred_I = processed_model1.predict_classes(eval_I)

pred_3 = processed_model1.predict_classes(eval_3)

pred_4 = processed_model1.predict_classes(eval_4)



#assign the predictions to the submission data

submission.iloc[0:100000, 1] = pred_A

submission.iloc[300000:400000, 1] = pred_D

submission.iloc[800000:900000, 1] = pred_I

submission.iloc[1000000:1500000, 1] = pred_3

submission.iloc[1500000:2000000, 1] = pred_4
#prediction of the subpart of E



eval_E = eval_data.signal.values[400000:500000].reshape(-1,1)

eval_E = np.expand_dims(eval_E, axis = 2)



pred_E = processed_model2.predict_classes(eval_E)



#assign the prediction to the submission data

submission.iloc[400000:500000, 1] = pred_E
#prediction of the subparts of B and J



eval_B = eval_data.signal.values[100000:200000].reshape(-1,1)

eval_B = np.expand_dims(eval_B, axis = 2)



eval_J = eval_data.signal.values[900000:1000000].reshape(-1,1)

eval_J = np.expand_dims(eval_J, axis = 2)



pred_B = processed_model3.predict_classes(eval_B)

pred_J = processed_model3.predict_classes(eval_J)



#assign the predictions to the submission data

submission.iloc[100000:200000, 1] = pred_B

submission.iloc[900000:1000000, 1] = pred_J
#prediction of the subparts of C and G



eval_C = eval_data.signal.values[200000:300000].reshape(-1,1)

eval_C = np.expand_dims(eval_C, axis = 2)



eval_G = eval_data.signal.values[600000:700000].reshape(-1,1)

eval_G = np.expand_dims(eval_G, axis = 2)



pred_C = processed_model4.predict_classes(eval_C)

pred_G = processed_model4.predict_classes(eval_G)



#assign the predictions to the submission data

submission.iloc[200000:300000, 1] = pred_C

submission.iloc[600000:700000, 1] = pred_G
#prediction of the subparts of F and H



eval_F = eval_data.signal.values[500000:600000].reshape(-1,1)

eval_F = np.expand_dims(eval_F, axis = 2)



eval_H = eval_data.signal.values[700000:800000].reshape(-1,1)

eval_H = np.expand_dims(eval_H, axis = 2)



pred_F = processed_model5.predict_classes(eval_F)

pred_H = processed_model5.predict_classes(eval_H)



#assign the predictions to the submission data

submission.iloc[500000:600000, 1] = pred_F

submission.iloc[700000:800000, 1] = pred_H
plt.figure(figsize=(20,5))

res = 1000

plt.plot(range(0,eval_data.shape[0],res),submission.open_channels[0::res])

for i in range(5): plt.plot([i*500000,i*500000],[-5,12.5],'r')

for i in range(21): plt.plot([i*100000,i*100000],[-5,12.5],'r:')

for k in range(4): plt.text(k*500000+250000,10,str(k+1),size=20)

for k in range(10): plt.text(k*100000+40000,7.5,let[k],size=16)

plt.title('Test Data Predictions',size=16)

plt.show()
submission.to_csv('ion_switch_submission.csv', index = False, float_format='%.4f')