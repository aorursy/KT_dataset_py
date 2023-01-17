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
data = pd.read_csv('/kaggle/input/pepsi-projesi/testtrain.csv')
data.head()
data.tail()
#data.sort_values(['ID'], axis=0, ascending=True, inplace=True) 

months = ['2017_Ocak', '2017_Şubat', '2017_Mart', '2017_Nisan', '2017_Mayıs', '2017_Haziran', 

          '2017_Temmuz', '2017_Ağustos', '2017_Eylül', '2017_Ekim', '2017_Kasım', '2017_Aralık',

         '2018_Ocak', '2018_Şubat', '2018_Mart', '2018_Nisan', '2018_Mayıs', '2018_Haziran', 

          '2018_Temmuz', '2018_Ağustos', '2018_Eylül', '2018_Ekim', '2018_Kasım', '2018_Aralık',

         '2019_Ocak', '2019_Şubat', '2019_Mart', '2019_Nisan', '2019_Mayıs', '2019_Haziran', 

          '2019_Temmuz']





data['Months'] = pd.Categorical(data['Months'], months)
data['Months']
data = data.sort_values('Months')
data.head()
data.tail()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



data['Months'] = le.fit_transform(data.Months.values)
#train_data = data.iloc[:-2500, :]

#test_data = data.iloc[-2500:, :]
#train_data.shape
#train_data.tail()
#train_data['Months'] += 1

#test_data['Months'] += 1
#sample_train = train_data[train_data['ID'] == '1000']

#sample_train = train_data.loc[train_data['ID'] == 1000]

#sample_test = test_data.loc[test_data['ID'] == 1000]

#sample_data = data.loc[data['ID'] == 1000]
#create target

target = []



#create data frame for features

features = pd.DataFrame()



for consumer_id in range(1000, 1500):

    #take one sample data for one consumer

    sample_data = data.loc[data['ID'] == consumer_id]

    

    #take feature of KG_N

    temp_column = sample_data['KG_N']

    sample_data = sample_data.drop(['KG_N'], axis = 1)

    

    #we can take first row, because others are same

    sample_data = sample_data.head(1)

    

    #the data of KG_N 

    sale_list = list(temp_column)

    #sample_test = test_data.loc[test_data['ID'] == consumer_id]

    

    #data of last 5 months is ready for target value. Because, we try to forecast 

    # sale(KG_N) data for future 5 months 

    

    for sale_index in range(0,len(sale_list[:-5])):

        sample_data['KG_N' + str(sale_index+1)] = sale_list[sale_index]

    

    #target list is ready 

    sample_target = sale_list[-5:]

    target.append(sample_target)

    

    #we create a new data frame. After processing of one consumer, we add this to data frame  

    features = features.append(sample_data)

    

        
features.head()
#from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler(feature_range = (0, 1))



#features['Age_0btw14'] = scaler.fit_transform(features['Age_0btw14'])
features.tail()
features = features.drop(['Months', 'ID'], axis = 1)

features.head()
#train_target = sample_train['KG_N']

#train_features = sample_train.drop(['KG_N'], axis = 1)



#sample_features = sample_data.drop(['Age_0btw14', 'Age_15btw24', 'Age_25btw34', 

#                                    'Age_35_plus', 'SES_AB','SES_C', 'SES_DE'], axis = 1)



#features = sample_features.iloc[:-5]

#target = sample_features.iloc[-5:]



#test_target = sample_test['KG_N']

#test_features = sample_test.drop(['KG_N'], axis = 1)



#train_target = train_data['KG_N']

#train_features = train_data.drop(['KG_N'], axis = 1)

features.shape
target = np.array(target)

target.shape
#target = target['KG_N']
#train_features = np.expand_dims(train_features, axis = 2)

#test_features = np.expand_dims(test_features, axis = 2)

features = np.expand_dims(features, axis = 2)

#target = np.expand_dims(target, axis = 1)
features.shape
from sklearn.model_selection import train_test_split

train_features, test_features, train_target, test_target = train_test_split(features, target, test_size = 0.2)
train_features.shape
test_features.shape
from keras.layers import LSTM, Dense, Dropout, Flatten, BatchNormalization, Conv1D, MaxPooling1D

from keras.models import Sequential

from keras.regularizers import l1, l2, l1_l2
model = Sequential()
#model.add(Conv1D(25, 2, input_shape = (train_features.shape[1], 1), activation = 'relu'))

#model.add(MaxPooling1D(pool_size = 2))

#model.add(LSTM(120, return_sequences=True, activation='relu'))

model.add(LSTM(32, input_shape = (features.shape[1], 1), return_sequences=True, activation='relu'))



#model.add(Dropout(0.5))

model.add(LSTM(36, return_sequences=True, activation='tanh'))



model.add(LSTM(32, return_sequences=False, activation='relu'))

#model.add(Dropout(0.5))



#model.add(Dense(500, activation = 'relu'))

#model.add(Dense(32, activation = 'relu'))

#model.add(Dropout(0.5))

#model.add(Dense(64, activation = 'relu'))

#model.add(Dropout(0.5))

model.add(Dense(5, activation = 'linear'))
import keras 

#opt_param = keras.optimizers.Adam(learning_rate = 0.01)

model.compile(optimizer = 'Adam', loss = 'mean_absolute_error')
rnn = model.fit(train_features, train_target, batch_size=64, epochs=25, validation_data=(test_features, test_target))
import matplotlib.pyplot as plt



#Plot the Loss Curves

plt.figure(figsize=[8,6])

plt.plot(rnn.history['loss'],'r',linewidth=3.0)

plt.plot(rnn.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss'],fontsize=18)

plt.legend(['Test loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves',fontsize=16)
prediction = model.predict(features)
prediction.shape
prediction
result_df = pd.DataFrame(columns = ['ID-Months', 'KG_N'])
predicted_values = []

id_list = []





for consumer_id in range(1000, 1500):

    month_id_list = [0,4,2,1,3]

    for month_id in month_id_list:

        predicted_values.append(prediction[consumer_id-1000][month_id])

        

    month_list = ['2019_Ağustos', '2019_Aralık', '2019_Ekim', '2019_Eylül', '2019_Kasım']

    

    for month_name in month_list:

        id_list.append(str(consumer_id) + '-' + month_name)
result_df['ID-Months'] = id_list

result_df['KG_N'] = predicted_values
result_df.head()
result_df.tail()
result_df.to_csv('submission.csv', index = False, encoding = 'utf-8-sig')