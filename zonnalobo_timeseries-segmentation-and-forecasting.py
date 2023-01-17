import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
pd.set_option('display.max_columns', 500)
data_ori = pd.read_csv('../input/daily_electricity_usage.csv')
data_ori['date'] = pd.to_datetime(data_ori['date'])
data_ori.head()
data = pd.DataFrame({'date':pd.date_range('2009-07-14',periods=536,freq='D',)})
for i in range(1000,7445):
    S=data_ori[data_ori['Meter ID']==i][['date','total daily KW']]
    data=pd.merge(data,S,how='left',on='date')
for i in range(1,6446):
    data.columns.values[i]="ID"+str(999+i)
data.head()
data.isnull().sum().sum()
data = data.fillna(data.mean())
data.date = pd.to_datetime(data.date)
data['day'] = data['date'].apply(lambda x:x.weekday())
x_call = data.columns[1:-1]
data_fix = pd.DataFrame({'Meter ID':range(1000,7445,1),'total KW':np.sum(data[x_call]).values})
data_fix['average per day']=data[x_call].mean().values
data_fix['% Monday']=data[data['day']==0][x_call].sum().values/data_fix['total KW']*100
data_fix['% Tuesday']=data[data['day']==1][x_call].sum().values/data_fix['total KW']*100
data_fix['% Wednesday']=data[data['day']==2][x_call].sum().values/data_fix['total KW']*100
data_fix['% Thursday']=data[data['day']==3][x_call].sum().values/data_fix['total KW']*100
data_fix['% Friday']=data[data['day']==4][x_call].sum().values/data_fix['total KW']*100
data_fix['% Saturday']=data[data['day']==5][x_call].sum().values/data_fix['total KW']*100
data_fix['% Sunday']=data[data['day']==6][x_call].sum().values/data_fix['total KW']*100
data_fix['% weekday']=data[(data['day']!=5)&(data['day']!=6)][x_call].sum().values/data_fix['total KW']*100
data_fix['% weekend']=data[(data['day']==5)|(data['day']==6)][x_call].sum().values/data_fix['total KW']*100
data_fix=data_fix.fillna(0)
data_fix.head()
from sklearn.preprocessing import StandardScaler
x_calls = data_fix.columns[1:]
scaller = StandardScaler()
matrix = pd.DataFrame(scaller.fit_transform(data_fix[x_calls]),columns=x_calls)
matrix['Meter ID'] = data_fix['Meter ID']
print(matrix.head())
corr = matrix[x_calls].corr()
fig, ax = plt.subplots(figsize=(8, 6))
cax=ax.matshow(corr,vmin=-1,vmax=1)
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.xticks(rotation=90)
plt.colorbar(cax)
def plot_BIC(matrix,x_calls,K):
    from sklearn import mixture
    BIC=[]
    for k in K:
        model=mixture.GaussianMixture(n_components=k,init_params='kmeans')
        model.fit(matrix[x_calls])
        BIC.append(model.bic(matrix[x_calls]))
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(K,BIC,'-cx')
    plt.ylabel("BIC score")
    plt.xlabel("k")
    plt.title("BIC scoring for K-means cell's behaviour")
    return(BIC)
K = range(2,31)
BIC = plot_BIC(matrix,x_calls,K)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
cluster = KMeans(n_clusters=5,random_state=217)
matrix['cluster'] = cluster.fit_predict(matrix[x_calls])
print(matrix.cluster.value_counts())
d=pd.DataFrame(matrix.cluster.value_counts())
fig, ax = plt.subplots(figsize=(8, 6))
plt.bar(d.index,d['cluster'],align='center',alpha=0.5)
plt.xlabel('Cluster')
plt.ylabel('number of data')
plt.title('Cluster of Data')
from sklearn.metrics.pairwise import euclidean_distances
distance = euclidean_distances(cluster.cluster_centers_, cluster.cluster_centers_)
print(distance)
# Reduction dimention of the data using PCA
pca = PCA(n_components=3)
matrix['x'] = pca.fit_transform(matrix[x_calls])[:,0]
matrix['y'] = pca.fit_transform(matrix[x_calls])[:,1]
matrix['z'] = pca.fit_transform(matrix[x_calls])[:,2]

# Getting the center of each cluster for plotting
cluster_centers = pca.transform(cluster.cluster_centers_)
cluster_centers = pd.DataFrame(cluster_centers, columns=['x', 'y', 'z'])
cluster_centers['cluster'] = range(0, len(cluster_centers))
print(cluster_centers)
# Plotting for 2-dimention
fig, ax = plt.subplots(figsize=(8, 6))
scatter=ax.scatter(matrix['x'],matrix['y'],c=matrix['cluster'],s=21,cmap=plt.cm.Set1_r)
ax.scatter(cluster_centers['x'],cluster_centers['y'],s=70,c='blue',marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)
plt.title('Data Segmentation')
# Plotting for 3-Dimention
fig, ax = plt.subplots(figsize=(8, 6))
ax=fig.add_subplot(111, projection='3d')
scatter=ax.scatter(matrix['x'],matrix['y'],matrix['z'],c=matrix['cluster'],s=21,cmap=plt.cm.Set1_r)
ax.scatter(cluster_centers['x'],cluster_centers['y'],cluster_centers['z'],s=70,c='red',marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.colorbar(scatter)
plt.title('Data Segmentation')
data_fix['cluster']=matrix['cluster']
print(data_fix[data_fix.columns[1:]].groupby(['cluster']).agg([np.mean]))
list(data_fix[data_fix.cluster==2]['Meter ID'])
data_cluster=data_fix[['Meter ID','cluster']]
data_forc=pd.DataFrame({'ds':pd.to_datetime(data['date'])})
for k in range(len(cluster_centers)):
    data_clus=data_cluster[data_cluster['cluster']==k]
    del data_clus['cluster']
    s1="cluster "+str(k)
    data_forc[s1]=0
    for i in list(data_clus.iloc[:,0]):
        s2="ID"+str(i)
        data_forc[s1]+=data[s2]
data_forc=data_forc.fillna(0)
data_forc_0=data_forc[['ds','cluster 0']]
data_forc_0.columns=['ds','y']

data_forc_1=data_forc[['ds','cluster 1']]
data_forc_1.columns=['ds','y']

data_forc_2=data_forc[['ds','cluster 2']]
data_forc_2.columns=['ds','y']

data_forc_3=data_forc[['ds','cluster 3']]
data_forc_3.columns=['ds','y']

data_forc_4=data_forc[['ds','cluster 4']]
data_forc_4.columns=['ds','y']

data_forc_all=pd.DataFrame({'ds':data_forc['ds']})
data_forc_all['y']=data_forc['cluster 0']+data_forc['cluster 1']+data_forc['cluster 2']+data_forc['cluster 3']+data_forc['cluster 4']
def plot_data(data_forc):
    timeseries=data_forc.copy()
    timeseries.columns=['date','Total Daily KW']
    timeseries = timeseries.set_index('date') 
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(timeseries.index,timeseries['Total Daily KW'],c='black',s=2)
import fbprophet
from sklearn.metrics import mean_squared_error, r2_score
def predic_fbp(data_forc,n_days):
    ny=pd.DataFrame({'holiday':"New Year's Day",'ds':pd.to_datetime(['2010-01-01','2011-01-01','2012-01-01']),
                     'lower_window':-1,'upper_window':1,})
    ch=pd.DataFrame({'holiday':"Christmas",'ds':pd.to_datetime(['2009-12-25','2010-12-25','2011-12-25','2012-12-25']),
                     'lower_window':0,'upper_window':1,})
    holidays=pd.concat([ny,ch])
    model = fbprophet.Prophet(daily_seasonality=False,weekly_seasonality=True,
                yearly_seasonality=True,changepoint_prior_scale=0.05,changepoints=None,
                holidays=holidays,interval_width=0.95)
    model.add_seasonality(name='monthly',period=30.5,fourier_order=5)
    size = len(data_forc) - n_days
    train, test = data_forc[0:size], data_forc[size:]
    test_=test.set_index('ds')
    model.fit(train)
    predics=model.predict(data_forc)
    test=pd.merge(test,predics[['ds','yhat','yhat_lower','yhat_upper']],how='left',on='ds')
    train=pd.merge(train,predics[['ds','yhat','yhat_lower','yhat_upper']],how='left',on='ds')
    RMSE=np.sqrt(mean_squared_error(test['y'], test['yhat']))
    print('RMSE = %.2f' % RMSE)
    R2=r2_score(test['y'], test['yhat'])
    print('R Square = %.2f'% R2)
    future = model.make_future_dataframe(periods=365+n_days, freq='D')
    future=model.predict(future)
    fig=model.plot(predics)
    plt.scatter(test_.index,test_['y'],c='black',s=7)
    fig2=model.plot(future)
    plt.scatter(test_.index,test_['y'],c='black',s=7)
    fig3=model.plot_components(future)
    return(train,test,predics,future,RMSE,R2)
plot_data(data_forc_0)
train_0,test_0,predics_0,future_0,RMSE_0,R2_0=predic_fbp(data_forc_0,90)
plot_data(data_forc_1)
train_1,test_1,predics_1,future_1,RMSE_1,R2_1=predic_fbp(data_forc_1,90)
plot_data(data_forc_3)
train_3,test_3,predics_3,future_3,RMSE_3,R2_3=predic_fbp(data_forc_3,90)
plot_data(data_forc_4)
train_4,test_4,predics_4,future_4,RMSE_4,R2_4=predic_fbp(data_forc_4,90)
plot_data(data_forc_all)
train_all,test_all,predics_all,future_all,RMSE_all,R2_all=predic_fbp(data_forc_all,90)