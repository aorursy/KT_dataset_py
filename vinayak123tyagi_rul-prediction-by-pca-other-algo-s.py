# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import cm



from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import QuantileTransformer , PowerTransformer





from keras.layers import Dense , LSTM

from keras.models import Sequential

from sklearn.metrics import mean_squared_error



import warnings 

warnings.filterwarnings('ignore')



%matplotlib inline

cmap = cm.get_cmap('Spectral') # Colour map (there are many others)



from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import explained_variance_score

from sklearn.metrics import r2_score
# loading triain_FD001 file

train_file = "/kaggle/input/nasa-cmaps/CMaps/train_FD001.txt" 

test_file = "/kaggle/input/nasa-cmaps/CMaps/test_FD001.txt"

RUL_file = "/kaggle/input/nasa-cmaps/CMaps/RUL_FD001.txt"



df = pd.read_csv(train_file,sep=" ",header=None)

df.head()
df.drop(columns=[26,27],inplace=True)
columns = ["Section-{}".format(i)  for i in range(26)]

df.columns = columns

df.head()
# Names 

MachineID_name = ["Section-0"]

RUL_name = ["Section-1"]

OS_name = ["Section-{}".format(i) for i in range(2,5)]

Sensor_name = ["Section-{}".format(i) for i in range(5,26)]



# Data in pandas DataFrame

MachineID_data = df[MachineID_name]

RUL_data = df[RUL_name]

OS_data = df[OS_name]

Sensor_data = df[Sensor_name]



# Data in pandas Series

MachineID_series = df["Section-0"]

RUL_series = df["Section-1"]
MachineID_series.unique()
grp = RUL_data.groupby(MachineID_series)

max_cycles = np.array([max(grp.get_group(i)["Section-1"]) for i in MachineID_series.unique()])

max_cycles
print("Max Life >> ",max(max_cycles))

print("Mean Life >> ",np.mean(max_cycles))

print("Min Life >> ",min(max_cycles))
df.info()
df.describe()
# Now, Initially visulizing the Operation Settings 2,3,4

df.plot(x=RUL_name[0], y= OS_name[0], c='k'); df.plot(x=RUL_name[0], y=OS_name[0], kind= "kde") 

df.plot(x=RUL_name[0], y=OS_name[1], c='k'); df.plot(x=RUL_name[0], y=OS_name[1], kind='kde')

df.plot(x=RUL_name[0], y=OS_name[2], c='k')#; df.plot(x=RUL_name[0], y=OS_name[2], kind='kde') 
for name in Sensor_name:

    df.plot(x=RUL_name[0], y=name, c='k')
data = pd.concat([RUL_data,OS_data,Sensor_data], axis=1)

data.drop(data[["Section-4", # Operatinal Setting

                "Section-5", # Sensor data

                "Section-9", # Sensor data

                "Section-10", # Sensor data

                "Section-14",# Sensor data

                "Section-20",# Sensor data

                "Section-22",# Sensor data

                "Section-23"]], axis=1 , inplace=True)
def Normalize(dataframe):

    gen = MinMaxScaler(feature_range=(0, 1))

    gen_data = gen.fit_transform(dataframe)

    return gen_data



def reshaping(train_X, train_y):

    abc = np.array(train_X).reshape(-1,1)

    asdsad= np.reshape(abc, (abc.shape[0], 1, abc.shape[1]))

    return asdsad, np.array(train_y)
# Making the funtion which Outputs RUL Dataframe

def RUL_df():

    rul_lst = [j  for i in MachineID_series.unique() for j in np.array(grp.get_group(i)[::-1]["Section-1"])]

    rul_col = pd.DataFrame({"rul":rul_lst})

    return rul_col



RUL_df().head()
#  Now, getting the data & Split it 

normalize_labels = Normalize(np.array(RUL_df()).reshape(-1,1)).reshape(1,-1)[0]

# print(normalize_labels.reshape(1,-1)[0])



train_X , test_X , train_Y , test_Y = train_test_split(Normalize(data),normalize_labels , test_size = 0.2)
#### Moving towards the model making and it's requiste

def lstm_reshaping(train_X, train_y):

    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))

    train_Y = np.array(train_y).reshape(1,-1)[0]

    return train_X, train_Y
# Making data according to model input requirements:

trainX, trainY = lstm_reshaping(train_X, train_Y)

testX, testY = lstm_reshaping(test_X, test_Y)



# Examining

print(trainX.shape)

print(testX.shape)
look_back = 17

# Model

model = Sequential()

model.add(LSTM(32, input_shape=(1, look_back)))

model.add(Dense(1))



model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
model.summary()
history = model.fit(trainX, trainY, epochs=10, batch_size=4, validation_data=(testX, testY))
print(history.history.keys())
# Accuracy Graph

plt.plot(history.epoch, history.history['accuracy'] , label="accuracy")

plt.plot(history.epoch, history.history['val_accuracy'] , label = "val_accuracy")

plt.legend()

plt.show()
# Loss Graph

plt.plot(history.epoch, history.history['loss'] , label = "loss")

plt.plot(history.epoch, history.history['val_loss'] , label = "val_loss")

plt.legend()

plt.show()
# Now Making the Prediction and Evaluating the model

score = model.evaluate(testX, testY, verbose = 0)

print("%s: %.2f%%" % ("acc", score[1]*100))
# Checking the model prediction on graphical form 

print(len(testY))

print(len(testX))



plt.plot(testY, label = "Actual")

plt.xlabel("Data");plt.ylabel("RUL")



val = model.predict(testX)

plst = [i[0] for i in val]

plt.plot(plst, c='k', label="Predict")

plt.legend();plt.show()
def rolling_mean(pandas_df):

    data = pandas_df.rolling(20).mean()

    return data



def method_PCA(df):

    pca = PCA(n_components=1)

    data = pca.fit_transform(Normalize(df))

    return data



def transform_data(data):

    pt = PowerTransformer()

    transform_data = pt.fit_transform(data)

    return transform_data 
# grouping w.r.t MID (Machine ID)

col_names = data.columns

def grouping(datafile, mid_series):

    data = [x for x in datafile.groupby(mid_series)]

    return data  
# APPLYING PCA WITH RESPECT TO THERE MID DATA

def data_processing(dataframe, grp_mid_series):

    pca_data = grouping(dataframe, grp_mid_series)

    process_lst_of_lst =[]

    jk =1

    for i in pca_data:

        dfs = i[1] 



        time = dfs["Section-1"]

        data = method_PCA(dfs)



        print("----------------MachineID-{}----------------".format(jk))



        data = transform_data(data)

        process_lst_of_lst.append(data)



        plt.plot(time, data)

        plt.show()



        jk = jk+1

    return process_lst_of_lst



process_lst_of_lst = data_processing(df, MachineID_series)
### Visulization of all 100 machine data all togther after applying PCA

for i in process_lst_of_lst:

    val = [j for j in range(len(i))]

    plt.plot(val,i)
process_data_lst = [j for i in process_lst_of_lst for j in i.reshape(1,-1)[0]]

process_df = pd.DataFrame({"MID":MachineID_series, 'data':process_data_lst, 'rul':list(RUL_df()["rul"])})

process_df.head()
train_X , test_X , train_y , test_y = train_test_split(process_df['data'], process_df['rul'] , test_size = 0.01)
# Making data according to model input requirements:

trainX, trainY = reshaping(train_X, train_y)

testX, testY = reshaping(test_X, test_y)



# Examining

print(trainX.shape)

print(testX.shape)
look_back = 1

# Model

model = Sequential()

model.add(LSTM(50, input_shape=(1, look_back)))

model.add(Dense(1))



model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'mse'])
model.summary()
history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_data=(testX, testY))
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
print(history.history.keys())
plt.plot(history.epoch, history.history['mae'] , label="mae")

plt.plot(history.epoch, history.history['val_mae'] , label = "val_mae")

plt.legend()

plt.show()
plt.plot(history.epoch, history.history['mse'] , label="mse")

plt.plot(history.epoch, history.history['val_mse'] , label = "val_mse")

plt.legend()

plt.show()
print(len(testY))

print(len(testX))



plt.plot(testY, label = "Actual")

plt.xlabel("Data");plt.ylabel("RUL")

val = model.predict(testX)

plst = [i[0] for i in val]

plt.plot(plst, c='k', label="Predict")

plt.legend();plt.show()
process_df.head()
from sklearn.linear_model import LinearRegression

X = np.array(process_df['data']).reshape(-1,1)



y = np.array(process_df['rul']).reshape(-1,1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 



reg = LinearRegression()

reg.fit(X_train, y_train)
## Accuracy of Linear Regression

print("Acc of lR >> ",reg.score(X_test, y_test))
plt.scatter(X_train, y_train, c='k', label='training data')

plt.scatter(X_test, y_test, label='testing data') 

pred = reg.predict(X_test)

plt.plot(X_test, pred, c='r' , label='regression line')

plt.xlabel("Sensor Data"); plt.ylabel("RUL values")

plt.legend()

plt.show()
# Now, trying to check this in all machine-id

def graphs(model_instance):

    for i in range(1,101):

        dataset = process_df.groupby('MID').get_group(i)

        print("MID-{}".format(i));

        testX = np.array(dataset['data']).reshape(-1,1)

        testy = np.array(dataset['rul']).reshape(-1,1)



        # score

        print("acc at MID-{} >> ".format(i), model_instance.score(testX, testy))



        # graph

        pred = model_instance.predict(testX)

        plt.scatter(testX, testy) 

        plt.plot(testX, pred, c='r')

        plt.show()



graphs(reg)
print("R2 Score >>", r2_score(y_test, pred))

print("explained_variance_score >> ", explained_variance_score(y_test, pred))

print("mean_squared_error >> ", mean_squared_error(y_test, pred))

print("mean_absolute_error >>",mean_absolute_error(y_test, pred))

# print('cv score >> ',cross_val_score(reg, testX, testy, cv=3))
plt.plot(y_test)

plt.plot(pred,c='k')
# trying dession tree regreesor

from sklearn.tree import DecisionTreeRegressor



dt_reg = DecisionTreeRegressor()

dt_reg.fit(X_train, y_train)

print("acc of DTS >> ",dt_reg.score(X_test, y_test))
plt.scatter(X_train, y_train, c='k', label='training data')

plt.scatter(X_test, y_test, label='testing data') 

dt_pred = dt_reg.predict(X_test)

plt.scatter(X_test, dt_pred, c='r' , label='prediction')

plt.xlabel("Sensor Data"); plt.ylabel("RUL values")

plt.legend()

plt.show()
print("R2 Score >>", r2_score(y_test, dt_pred))

print("explained_variance_score >> ", explained_variance_score(y_test, dt_pred))

print("mean_squared_error >> ", mean_squared_error(y_test, dt_pred))

print("mean_absolute_error >>",mean_absolute_error(y_test, dt_pred))

# print('cv score >> ',cross_val_score(dt_reg, testX, testy, cv=3))
# Now, trying to check this in all machine-id        

graphs(dt_reg)
plt.plot(y_test)

plt.plot(dt_pred,c='k')
df_test = pd.read_csv(test_file, sep=" ",header=None)

df_test.drop(columns=[26,27],inplace=True)

df_test.columns = columns

df_test.head()
df_rul = pd.read_csv(RUL_file, names=['rul'])

df_rul.head()
df_test.drop(df_test[["Section-4", # Operatinal Setting

                "Section-5", # Sensor data

                "Section-9", # Sensor data

                "Section-10", # Sensor data

                "Section-14",# Sensor data

                "Section-20",# Sensor data

                "Section-22",# Sensor data

                "Section-23"]], axis=1 , inplace=True)



test_mid = df_test["Section-0"]

df_test.head()
test_lst=data_processing(df_test,test_mid)
test_process_data_lst = [j for i in test_lst for j in i.reshape(1,-1)[0]]

test_process_df = pd.DataFrame({"MID":test_mid, 'data':test_process_data_lst})

test_process_df.head()
val = test_process_df.groupby('MID')

val = [np.array(val.get_group(i)['data']).reshape(-1,1) for i in range(1,101)]

# print(val)

# print(np.array(test_rul_csv)[2][0])
# lr prediction

count = 0

rul = np.array(df_rul)

for mid_val in val:

    lr_predict = reg.predict(mid_val)

    plt.plot([i for i in range(len(mid_val))], [rul[count][0] for i in range(len(mid_val))], c='r')

    plt.plot(lr_predict)

    count = count +1

    plt.show()