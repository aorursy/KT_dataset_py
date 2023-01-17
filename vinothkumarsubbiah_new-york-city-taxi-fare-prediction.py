import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
data = pd.read_csv(r'../input/train.csv', nrows= 800000)

data.head(5)
data.shape
data.describe()
train_df = data[(data.fare_amount > 0)] 

train_df = train_df[(train_df.passenger_count > 0)]
#Check missing values

train_df.isnull().sum()
#New york city Latitude and longitude coordinates are: 40.730610, -73.935242.Boundary of New York City is (-75, -73, 40, 42)

train_df = train_df[(train_df['pickup_longitude'] > -75) & (train_df['pickup_longitude'] <= -73)]

train_df = train_df[(train_df['pickup_latitude'] > 40) & (train_df['pickup_latitude'] <= 42)]

train_df = train_df[(train_df['dropoff_longitude'] > -75) & (train_df['dropoff_longitude'] <= -73)]

train_df = train_df[(train_df['dropoff_latitude'] > 40) & (train_df['dropoff_latitude'] <= 42)]



#Remove same pickup and dropoff latitude and longitude

train_df = train_df[(train_df['pickup_latitude'] != train_df['dropoff_latitude']) & (train_df['pickup_longitude'] != train_df['dropoff_longitude'])]
train_df.shape
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])

def split_datetime(df):

    train_df['year'] = train_df.pickup_datetime.dt.year

    train_df['month'] = train_df.pickup_datetime.dt.month 

    train_df['day'] = train_df.pickup_datetime.dt.day

    train_df['hour'] = train_df.pickup_datetime.dt.hour

    train_df['day_of_week'] = train_df.pickup_datetime.dt.dayofweek    

    return train_df
train_df = split_datetime(train_df)

train_df.drop('pickup_datetime',axis=1)

train_df.head(5)
pick_pos = train_df[['pickup_longitude','pickup_latitude']]

wcss=[]

for n_clusters in range(1,10):

    kmeans=KMeans(n_clusters=n_clusters,init='k-means++',)

    kmeans.fit(pick_pos)

    wcss.append(kmeans.inertia_)

    clusters = kmeans.predict(pick_pos)

plt.plot(range(1,10),wcss)

plt.title('Elbow Method')

plt.xlabel('No. of cluster')

plt.ylabel('wcss: sum of dist. of sample to their closest cluster center' )
drop_pos = train_df[['dropoff_longitude','dropoff_latitude']]

wcss=[]

for n_clusters in range(1,10):

    kmeans=KMeans(n_clusters=n_clusters,init='k-means++',)

    kmeans.fit(drop_pos)

    wcss.append(kmeans.inertia_)

    clusters = kmeans.predict(drop_pos)

plt.plot(range(1,10),wcss)

plt.title('Elbow Method')

plt.xlabel('No. of cluster')

plt.ylabel('wcss: sum of dist. of sample to their closest cluster center' )
model = KMeans(n_clusters = 6)

model.fit(pick_pos)

pcluster = model.predict(pick_pos)

cluster_p = pd.get_dummies(pcluster,prefix='pick_clust',drop_first=False)

train = pd.concat([train_df,cluster_p],axis=1)
train.shape
model = KMeans(n_clusters = 6)

model.fit(drop_pos)

dcluster = model.predict(drop_pos)

cluster_d = pd.get_dummies(pcluster,prefix='drop_clust',drop_first=False)

train = pd.concat([train,cluster_d],axis=1)

train.head(5)
train=train.dropna(how = 'any', axis = 'rows')
X = train.drop(['key','fare_amount','pickup_datetime'],axis=1)

y = train[['fare_amount']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rfr = RandomForestRegressor(n_estimators=30, max_features=20, max_depth=25, min_samples_split=2, min_samples_leaf=3, random_state=0)

rfr.fit(X_train, y_train)

pred = rfr.predict(X_test)

error = np.sqrt(mean_squared_error(y_test,pred))

print(error)
# Train Model with entire Data

rfr.fit(X, y)
predict_data = pd.read_csv(r'../input/test.csv')

predict_data.dropna(how = 'any', axis = 'rows')

predict_data.shape
predict_data.describe()
tpick_pos = predict_data[['pickup_longitude','pickup_latitude']]

kmodel = KMeans(n_clusters = 6)

kmodel.fit(tpick_pos)

pre_pcluster = kmodel.predict(tpick_pos)

pre_cluster_p = pd.get_dummies(pre_pcluster,prefix='pick_clust',drop_first=False)

predict_data = pd.concat([predict_data,pre_cluster_p],axis=1)

predict_data.shape
tdrop_pos = predict_data[['dropoff_longitude','dropoff_latitude']]

kmodel = KMeans(n_clusters = 6)

kmodel.fit(tdrop_pos)

pre_dcluster = kmodel.predict(tdrop_pos)

pre_cluster_d = pd.get_dummies(pre_dcluster,prefix='drop_clust',drop_first=False)

predict_data = pd.concat([predict_data,pre_cluster_d],axis=1)

predict_data.shape
predict_data.head(5)
predict_data['pickup_datetime'] = pd.to_datetime(predict_data['pickup_datetime'])

def split_datetime(pred_df):

    pred_df['year'] = pred_df.pickup_datetime.dt.year

    pred_df['month'] = pred_df.pickup_datetime.dt.month 

    pred_df['day'] = pred_df.pickup_datetime.dt.day

    pred_df['hour'] = pred_df.pickup_datetime.dt.hour

    pred_df['day_of_week'] = pred_df.pickup_datetime.dt.dayofweek    

    return pred_df
predict_data = split_datetime(predict_data)

predict_data.drop('pickup_datetime',axis=1)

predict_data.head(5)
final_data = predict_data.drop(['key','pickup_datetime'],axis=1)

final_data.shape
final_prediction = rfr.predict(final_data)
predict_data['fare_amount'] = final_prediction

predict_data.to_csv('submission.csv',index=False)

predict_data.head(5)