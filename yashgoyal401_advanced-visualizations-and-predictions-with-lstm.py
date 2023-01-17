import warnings

warnings.filterwarnings("ignore")
import pandas as pd

data = pd.read_csv("../input/london-bike-sharing-dataset/london_merged.csv")
data.head()
import tensorflow.compat.v1 as tf

print(tf.test.gpu_device_name())

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth

config = tf.ConfigProto()

config.gpu_options.allow_growth = True
import pandas as pd

import numpy as np



data.shape

data.info(memory_usage="deep")



import seaborn as sns

import matplotlib.pyplot as plt

sns.set(context="notebook", style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True)



data.columns

data.head()



data["timestamp"] = pd.to_datetime(data["timestamp"])



data = data.set_index("timestamp")



data["hour"] = data.index.hour

data["day_of_month"] = data.index.day

data["day_of_week"]  = data.index.dayofweek

data["month"] = data.index.month



data.columns

data.shape



corr_matrix = data.corr().abs()

high_corr_var=np.where(corr_matrix>0.8)

high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
print(high_corr_var)
data.head()
plt.figure(figsize=(16,6))

sns.heatmap(data.corr(),cmap="YlGnBu",square=True,linewidths=.5,center=0,linecolor="red")
plt.figure(figsize=(16,6))

data.isnull().sum()

sns.heatmap(data.isnull(),cmap="viridis")
plt.figure(figsize=(15,6))

sns.lineplot(data=data,x=data.index,y=data.cnt)

plt.xticks(rotation=90)
df_by_month = data.resample("M").sum()



plt.figure(figsize=(16,6))

sns.lineplot(data=df_by_month,x=df_by_month.index,y=df_by_month.cnt,color="red")

plt.xticks(rotation=90)
plt.figure(figsize=(16,6))

sns.pointplot(data=data,x=data.hour,y=data.cnt,color="black")
plt.figure(figsize=(16,6))

sns.pointplot(data=data,x=data.month,y=data.cnt,color="red")
plt.figure(figsize=(16,6))

sns.pointplot(data=data,x=data.day_of_week,y=data.cnt,color="black")
plt.figure(figsize=(16,6))

sns.lineplot(data=data,x=data.day_of_month,y=data.cnt,color="r")
plt.figure(figsize=(16,6))

sns.pointplot(data=data,x=data.hour,y=data.cnt,hue=data.is_holiday)
plt.figure(figsize=(16,6))

sns.pointplot(data=data,x=data.month,y=data.cnt,hue=data.is_holiday)
plt.figure(figsize=(16,6))

sns.pointplot(data=data,hue=data.season,y=data.cnt,x=data.month)
plt.figure(figsize=(16,6))

sns.countplot(data=data,hue=data.is_holiday,x=data.season)
plt.figure(figsize=(16,6))

sns.pointplot(data=data,x=data.hum,y=data.cnt,color="black")

plt.xticks(rotation=90)
plt.figure(figsize=(16,6))

sns.pointplot(data=data,x=data.wind_speed,y=data.cnt)

plt.xticks(rotation=90)
plt.figure(figsize=(16,6))

sns.pointplot(data=data,x=data.weather_code,y=data.cnt)

plt.xticks(rotation=90)
plt.figure(figsize=(16,6))

sns.lineplot(x=data.hour,y=data.cnt,data=data,hue=data.is_weekend)
plt.figure(figsize=(16,6))

sns.pointplot(x=data.hour,y=data.cnt,data=data,hue=data.season)
plt.figure(figsize=(16,6))

sns.pointplot(x=data.hour,y=data.cnt,data=data,hue=data.weather_code)
plt.figure(figsize=(16,6))

sns.countplot(data=data,x=data.day_of_week,hue=data.weather_code,palette="viridis")

plt.legend(loc="best")
plt.figure(figsize=(16,6))

sns.boxplot(data=data,x=data["hour"],y=data.cnt)
plt.figure(figsize=(16,6))

sns.boxplot(data=data,x=data["day_of_week"],y=data.cnt)
plt.figure(figsize=(16,6))

sns.boxplot(data=data,x=data["day_of_month"],y=data.cnt)
plt.figure(figsize=(16,6))

sns.boxplot(data=data,x=data["month"],y=data.cnt)
plt.figure(figsize=(16,6))

sns.boxplot(data=data,x=data["day_of_month"],y=data.cnt,hue=data["is_holiday"])
from sklearn.model_selection import train_test_split

train,test = train_test_split(data,test_size=0.1,random_state=0)



print(train.shape)

print(test.shape)

from sklearn.preprocessing import MinMaxScaler

scaler  = MinMaxScaler()



num_colu = ['t1', 't2', 'hum', 'wind_speed']

trans_1 = scaler.fit(train[num_colu].to_numpy())

train.loc[:,num_colu] = trans_1.transform(train[num_colu].to_numpy())

test.loc[:,num_colu] = trans_1.transform(test[num_colu].to_numpy())



cnt_scaler = MinMaxScaler()

trans_2 = cnt_scaler.fit(train[["cnt"]])

train["cnt"] = trans_2.transform(train[["cnt"]])

test["cnt"] = trans_2.transform(test[["cnt"]])
from tqdm import tqdm_notebook as tqdm

tqdm().pandas()

def prepare_data(X,y,time_steps=1):

    Xs = []

    Ys = []

    for i in tqdm(range(len(X) - time_steps)):

        a = X.iloc[i:(i + time_steps)].to_numpy()

        Xs.append(a)

        Ys.append(y.iloc[i+time_steps])

    return np.array(Xs),np.array(Ys)    



steps=24

X_train , y_train = prepare_data(train,train.cnt,time_steps=steps)

X_test , y_test = prepare_data(test,test.cnt,time_steps=steps)

print("X_train : {}\nX_test : {}\ny_train : {}\ny_test: {}".format(X_train.shape,X_test.shape,y_train.shape,y_test.shape))
### LSTMM model

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout , LSTM , Bidirectional 





model = Sequential()

model.add(Bidirectional(LSTM(128,input_shape=(X_train.shape[1],X_train.shape[2]))))

model.add(Dropout(0.2))

model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer="adam",loss="mse")



with tf.device('/GPU:0'):

    prepared_model = model.fit(X_train,y_train,batch_size=32,epochs=100,validation_data=[X_test,y_test])



plt.plot(prepared_model.history["loss"],label="loss")

plt.plot(prepared_model.history["val_loss"],label="val_loss")

plt.legend(loc="best")

plt.xlabel("No. Of Epochs")

plt.ylabel("mse score")
pred = model.predict(X_test)



y_test_inv = cnt_scaler.inverse_transform(y_test.reshape(-1,1))

pred_inv = cnt_scaler.inverse_transform(pred)



plt.figure(figsize=(16,6))

plt.plot(y_test_inv.flatten(),marker=".",label="actual")

plt.plot(pred_inv.flatten(),marker=".",label="prediction",color="r")
y_test_actual = cnt_scaler.inverse_transform(y_test.reshape(-1,1))

y_test_pred = cnt_scaler.inverse_transform(pred)



arr_1 = np.array(y_test_actual)

arr_2 = np.array(y_test_pred)



actual = pd.DataFrame(data=arr_1.flatten(),columns=["actual"])

predicted = pd.DataFrame(data=arr_2.flatten(),columns = ["predicted"])
final = pd.concat([actual,predicted],axis=1)

final.head()
from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(final.actual,final.predicted)) 

r2 = r2_score(final.actual,final.predicted) 

print("rmse is : {}\nr2 is : {}".format(rmse,r2))
plt.figure(figsize=(16,6))

plt.plot(final.actual,label="Actual data")

plt.plot(final.predicted,label="predicted values")

plt.legend(loc="best")