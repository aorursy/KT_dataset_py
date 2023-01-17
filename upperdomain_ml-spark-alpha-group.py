import os
import math
import numpy as np
import pandas as pd 
from mpl_toolkits import mplot3d
%matplotlib inline
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
#from  sklearn.linear_model import SGDRegressor
#from sklearn.neural_network import MLPRegressor
#from sklearn.ensemble import GradientBoostingRegressor
print(os.listdir("../input"))
#creating chunk of size 10^6 from training data for incremental training
def chunck_generator(filename, header=False,chunk_size = 10 ** 6):
    for chunk in pd.read_csv(filename,delimiter=',', iterator=True, chunksize=chunk_size, parse_dates=[1] ): 
        yield (chunk)
#this function will calculate distance 
alpha_ang = 0.506
def distance_travel(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()*50
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()*69
    df['displacement_vector'] = (df.abs_diff_latitude**2 + df.abs_diff_longitude**2)**0.5 ### as the crow flies  
    df['actual_long'] = (df.displacement_vector*np.sin(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude)-alpha_ang)).abs()
    df['actual_lat'] = (df.displacement_vector*np.cos(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude)-alpha_ang)).abs()
    df['distance_travel'] = df.actual_long + df.actual_lat
    return df
    

def data_clean(df):
    df=df[df.passenger_count>0]
    df.fare_amount = df.fare_amount.astype(np.float64)
    df=df[df.fare_amount>0]
    distance_travel(df)
    df=df[df.distance_travel>0]
    return df
    
def remove_outliers(df):
    df=df[df.distance_travel<30]
    df=df[df.fare_amount<60]
    return df
def graph_present(df):
    test=df[df.passenger_count==1]
    plot = test.iloc[:len(test)].plot.scatter('distance_travel','fare_amount')
    plot = df.iloc[:100000].plot.scatter('distance_travel','fare_amount')
#data pre-processing
def data_preprocessing(df):
    df=distance_travel(df)
    df=data_clean(df)
    df=remove_outliers(df)
    return df
#data analysis distance travel
df=pd.read_csv('../input/train.csv',nrows=10_00_000)
df=df=distance_travel(df)
df=df[df.passenger_count==1]
df=df[df.distance_travel<30]
#graph_present(df)
df.distance_travel.hist(bins=50, figsize=(12,4))
plt.xlabel('distance miles')
plt.title('Histogram ride distances in miles')
df.distance_travel.describe()
ax = plt.axes(projection='3d')
df=pd.read_csv('../input/train.csv',nrows=1_00_000)
df=distance_travel(df)
df=df[df.passenger_count<=6]
df=df[df.distance_travel<30]
#graph_present(df)
df=df[df.fare_amount>0]
df=df[df.fare_amount<60]
ax.scatter3D(df.passenger_count, df.distance_travel, df.fare_amount, c=df.distance_travel, cmap='Greens');
ax.set_xlabel("passenger_count")
ax.set_ylabel("distance_travel")
ax.set_zlabel("fare_amount")
#data analysis fare amount
df=pd.read_csv('../input/train.csv',nrows=1_00_000)
df=distance_travel(df)
df=df[df.passenger_count<=6]
df=df[df.distance_travel<30]
#graph_present(df)
df=df[df.fare_amount>0]
df=df[df.fare_amount<60]
df.fare_amount.hist(bins=50, figsize=(12,4))
plt.xlabel('fare_amount')
df.fare_amount.describe()
filename = r'../input/train.csv'
gen = chunck_generator(filename=filename)
#gbr_regr = GradientBoostingRegressor(n_estimators=100,warm_start=True) # incremental training
linear_regr=LinearRegression(copy_X=True,n_jobs=10)
#sgd_regr=SGDRegressor()
#mlp_regr=MLPRegressor(warm_start=True,verbose=True,learning_rate_init=0.03)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
t=1;
while t<=56:
    print("Chunk ",t )
    df=next(gen)
    df=data_preprocessing(df)
    l=len(df)
    df_train=df[:int(0.9*l)]
    df_test=df[int(0.9*l):]
    train_X = np.column_stack((df_train.distance_travel, df_train.passenger_count,df_train.pickup_longitude,df_train.dropoff_longitude, np.ones(len(df_train))))
    test_X = np.column_stack((df_test.distance_travel, df_test.passenger_count, df_test.pickup_latitude,df_test.dropoff_latitude, np.ones(len(df_test))))
    train_y = np.array(df_train.fare_amount)
    test_y=np.array(df_test.fare_amount)
    imp = imp.fit(train_X)
    linear_regr.fit(train_X, train_y)
    print("LinearRegressor",linear_regr.score(test_X,test_y))
 #   sgd_regressor.partial_fit(train_X, train_y)
  #  gbr_regr.fit(train_X, train_y)
 #   mlp_regr.fit(train_X, train_y)

#     print("SGDRegressor",sgd_regressor.score(test_X,test_y))
#     print("GradientBoostingRegressor",gbr_regr.score(test_X,test_y))
 #   print("MLPRegressor",mlp_regr.score(test_X,test_y))
    t+=1
regr=linear_regr    

# df=pd.read_csv('../input/train.csv',nrows=10_00_000)
# df=df[int(0.9*len(df)):]
# fig, ax = plt.subplots()
# distance_travel(df)
# df=df[df.passenger_count<=6]
# df=df[df.distance_travel<30]
# #graph_present(df)
# df=df[df.fare_amount>0]
# df=df[df.fare_amount<60]
# X = np.column_stack((df.distance_travel, df.passenger_count, np.ones(len(df))))
# pre=regr.predict(X)
# print("prediction success")
# ax.plot(df.key, df.fare_amount, label="y = x**2")
# ax.plot(df.key, pre, label="y = x**3")
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('title')
# ax.legend(loc=2);
test_df=pd.read_csv('../input/test.csv',nrows = 10_00_000)
distance_travel(test_df)
test_df.head()
test_X = np.column_stack((test_df.distance_travel, test_df.passenger_count,test_df.pickup_longitude,test_df.dropoff_longitude, np.ones(len(test_df))))
test_X = imp.transform(test_X)
predicted_fare=regr.predict(test_X)
print(predicted_fare)
print(np.mean(predicted_fare))
my_submission = pd.DataFrame({'key': test_df.key, 'fare_amount': predicted_fare})
my_submission.to_csv('submission.csv', index=False)
my_submission.head()