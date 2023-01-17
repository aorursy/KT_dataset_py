import numpy as np 

import pandas as pd 

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, LSTM, CuDNNLSTM, Dropout, BatchNormalization

from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from tensorflow.keras.callbacks import Callback, EarlyStopping



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import confusion_matrix

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import BorderlineSMOTE



import datetime as dt
def Load_LSTM_Data(data_dir, stock_file_list, resampling='BorderlineSMOTE2'):    

    print('Start loading LSTM data')

    t1 = dt.datetime.now()

    LSTM_X_data = np.zeros((1, time_steps, features), dtype=np.float32)

    LSTM_Y_data = np.zeros((1), dtype=np.int16)

    fail_list = []

    for stock in stock_file_list:

        try:

            #print('Processing symbol ', stock)

            numpy_data_X_file = stock + '-LSTM_X-timestep40-forward_day5-StandardScaler.npy'

            numpy_data_Y_file = stock + '-LSTM_Y-timestep40-forward_day5-StandardScaler.npy'

            data_X = np.load(data_dir + numpy_data_X_file)

            data_Y = np.load(data_dir + numpy_data_Y_file)        

            # Resample data to avoid imbalance classes

            data_X = data_X.reshape(data_X.shape[0], time_steps*features)

            if resampling=='BorderlineSMOTE2':

                data_X, data_Y = BorderlineSMOTE(kind='borderline-2').fit_resample(data_X, data_Y)

            data_X = data_X.reshape(data_X.shape[0], time_steps, features)            

            LSTM_X_data = np.append(LSTM_X_data, data_X, axis=0)

            LSTM_Y_data = np.append(LSTM_Y_data, data_Y, axis=0)

        except:

            print('*** Can not processing stock ', stock)

            fail_list.append(stock)       

    LSTM_X_data, LSTM_Y_data = shuffle(LSTM_X_data[1:], LSTM_Y_data[1:], random_state=42)

    t2 = dt.datetime.now()

    print('Failed list: ', fail_list)

    print('Loaded LSTM data. Finished in ', t2-t1)    

    return LSTM_X_data, LSTM_Y_data





feature_columns = ['Month', 'Day', 'DayofWeek', 'DayofYear',

           'Open', 'High', 'Low', 'Close', 'Volume',

           'MA10', 'MA20', 'MA50', 'MA100', 'MA200', 'MAV50', 'RSI14', 'MFI14', 'BB_top', 'BB_bot',

           'MACD_ml', 'MACD_sl', 'MACD_histogram', 'ADX', 'PDI', 'MDI', 'ATR', 'CCI',

           'Chaikin', 'OBV', 'ROC', 'StochasticMomentum', 'RS_Index', 'Accumulation_Distribution',

           'VNINDEX_O', 'VNINDEX_H', 'VNINDEX_L', 'VNINDEX_C', 'VNINDEX_V', 'VNINDEX_MA10', 'VNINDEX_MA20',

           'VNINDEX_MA50', 'VNINDEX_MA100', 'VNINDEX_MA200', 'VNINDEX_MAV50', 'VNINDEX_RSI14', 'VNINDEX_MFI14', 'VNINDEX_BB_top', 'VNINDEX_BB_bot',

           'DJI_O', 'DJI_H', 'DJI_L', 'DJI_C', 'DJI_V', 'DJI_MA50', 'DJI_MA100', 'DJI_MA200', 'DJI_MAV50',

           'SP500_O', 'SP500_H', 'SP500_L', 'SP500_C', 'SP500_V', 'SP500_MA50', 'SP500_MA100', 'SP500_MA200', 'SP500_MAV50']

time_steps = 40

features = len(feature_columns)

forward_day = 5

data_dir = '../input/vn30-vnindex-dji-sp500-5classes-standardscaler/'

# Stock symbol list for VN30

stock_file_list = ['CII', 'CTD', 'CTG', 'DHG', 'DPM', 'EIB', 'FPT', 'GAS', 'GMD', 'HDB', 'HPG', 'MBB', 'MSN', 

                   'MWG', 'NVL', 'PNJ', 'REE', 'ROS', 'SAB', 'SBT', 'SSI', 'STB', 'TCB', 'VCB', 'VHM', 'VIC', 

                   'VJC', 'VNM', 'VPB', 'VRE']



# Load shuffled LSTM data, without resampling

LSTM_X_data, LSTM_Y_data = Load_LSTM_Data(data_dir, stock_file_list, resampling='None')

#Check the shape of data

print('LSTM_X_data.shape: ', LSTM_X_data.shape)

print('LSTM_Y_data.shape: ', LSTM_Y_data.shape)



unique_origin, counts_origin = np.unique(LSTM_Y_data, return_counts=True)

data_dict = dict(zip(unique_origin, counts_origin))

print('Origin labels: ', data_dict)



# Plot 

#sns.countplot(LSTM_Y_data, palette=sns.color_palette("RdYlGn_r", 5))
def Create_LSTM_Model(unit_per_layer=1000, drop_out=0.5, optimizer='Adam', lr=1e-3):

    # Create model with LSTM & Dense layers

    model = Sequential()

    model.add(CuDNNLSTM(units=unit_per_layer, input_shape=(time_steps, features), return_sequences=True))

    model.add(Dropout(drop_out))

    model.add(BatchNormalization())

    model.add(CuDNNLSTM(units=unit_per_layer))     

    model.add(Dropout(drop_out))

    model.add(BatchNormalization())

    model.add(Dense(units=unit_per_layer, activation='tanh'))

    model.add(Dropout(drop_out))

    model.add(BatchNormalization())

    model.add(Dense(units=5, activation='softmax'))



    if optimizer.upper()=='ADAM':

        opti_func = Adam(lr=lr, amsgrad=True)

    elif optimizer.upper()=='SGD':

        opti_func = SGD(lr=lr)

    elif optimizer.upper()=='RMSPROP':

        opti_func = RMSprop(lr=lr)

              

    model.compile(optimizer=opti_func, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

   

    return model
model = Create_LSTM_Model(unit_per_layer=1000, drop_out=0.5, optimizer = 'Adam', lr=5e-4)

t1 = dt.datetime.now()

X_train, X_test, y_train, y_test = train_test_split(LSTM_X_data, LSTM_Y_data, test_size=0.1, random_state=42)

history = model.fit(X_train, y_train, epochs=30, batch_size=512, validation_data=(X_test, y_test))

t2 = dt.datetime.now()

print('Training time: ', t2-t1)  



model.evaluate(LSTM_X_data, LSTM_Y_data)
print('Origin labels: ', dict(zip(unique_origin, counts_origin)))

LSTM_y_hat = model.predict(LSTM_X_data)

LSTM_y_hat_pos = np.argmax(LSTM_y_hat, axis=1)

unique_predict, counts_predict = np.unique(LSTM_y_hat_pos, return_counts=True)

print('Predicted labels: ', dict(zip(unique_predict, counts_predict)))

confusion_m = confusion_matrix(LSTM_Y_data, LSTM_y_hat_pos)

print('Confusion matrix of predicted data:')

print(confusion_m)

for i in range(len(unique_predict)):

    print('Label {0} accuracy: {1:0.1f}%'.format(i, 100*confusion_m[i,i]/counts_origin[i]))
# Load shuffled LSTM data, and oversample when load each stock

LSTM_X_data_resample, LSTM_Y_data_resample = Load_LSTM_Data(data_dir, stock_file_list, resampling='BorderlineSMOTE2')

#Check the shape of data

print('LSTM_X_data_resample.shape: ', LSTM_X_data_resample.shape)

print('LSTM_Y_data_resample.shape: ', LSTM_Y_data_resample.shape)



unique_origin_resample, counts_origin_resample = np.unique(LSTM_Y_data_resample, return_counts=True)

data_dict_resample = dict(zip(unique_origin_resample, counts_origin_resample))

print('Origin labels: ', data_dict_resample)



# Plot 

#sns.countplot(LSTM_Y_data_resample, palette=sns.color_palette("RdYlGn_r", 5))
model_resample = Create_LSTM_Model(unit_per_layer=1000, drop_out=0.5, optimizer = 'Adam', lr=5e-4)

t1 = dt.datetime.now()

X_train_resample, X_test_resample, y_train_resample, y_test_resample = train_test_split(LSTM_X_data_resample, LSTM_Y_data_resample, 

                                                                                        test_size=0.1, random_state=42)

history = model_resample.fit(X_train_resample, y_train_resample, epochs=30, batch_size=512, 

                             validation_data=(X_test_resample, y_test_resample))



t2 = dt.datetime.now()

print('Training time: ', t2-t1)  



model_resample.evaluate(LSTM_X_data_resample, LSTM_Y_data_resample)



model_resample.evaluate(LSTM_X_data, LSTM_Y_data)
print('Origin labels: ', dict(zip(unique_origin, counts_origin)))

LSTM_y_hat2 = model_resample.predict(LSTM_X_data)

LSTM_y_hat_pos2 = np.argmax(LSTM_y_hat2, axis=1)

unique_predict2, counts_predict2 = np.unique(LSTM_y_hat_pos2, return_counts=True)

print('Predicted labels: ', dict(zip(unique_predict2, counts_predict2)))

confusion_m2 = confusion_matrix(LSTM_Y_data, LSTM_y_hat_pos2)

print('Confusion matrix of predicted data:')

print(confusion_m2)

for i in range(len(unique_predict2)):

    print('Label {0} accuracy: {1:0.1f}%'.format(i, 100*confusion_m2[i,i]/counts_origin[i]))