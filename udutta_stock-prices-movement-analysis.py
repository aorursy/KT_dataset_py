import pandas as pd

# pd.core.common.is_list_like = pd.api.types.is_list_like

from pandas_datareader import data

#helps in extracting data from included sources.  A very useful tool on pandas

from matplotlib import pyplot as plt

import datetime

import numpy as np

print(f'{np.__version__}: numpy version')

print(pd.__version__)

# print(f'{data.__version__} : pandas_datareader version')
# data.DataReader?
#removing SYMC as the data fetched in NAN, look later how to clean the data with SYMC



company_dict={

   'amazon':'AMZN','Apple':'AAPL','Walgreen':'WBA','NOrthrop Grunman':'NOC','Boeing':'BA','Lockhead Martin':'LMT','McDonalds':'MCD',

    'Intel':'INTC','Navister':'NAV','IBM':'IBM','Texas Instrument':'TXN','Mastercard':'MA','Microsoft':'MSFT','General ELectric':'GE',

    'Americal express':'AXP','Pepsi':'PEP','Coca Cola':'KO','Jhonson and Jhonson':'JNJ','Toyota':'TM','Mitsubishi':'MSBHY',

    'Exxon':'XOM','Valero ENgineering':'VLO','Bank of AMerica':'BAC','HOnda':'HMC','Sony':'SNE','Chevren':'CVX','Ford':'F'    

}

company_dict.items()
companies=sorted(company_dict.items(),key=lambda x:x[1])

companies

print(len(companies))
print(list(company_dict.values()),end=" ")

# Using print with end=" " to get a horizontal list
data_source='yahoo'

start_date='2016-01-01'

end_date='2018-12-31'

#use pandas_reader.data.DataReader 

panel_data=data.DataReader(list(company_dict.values()),data_source,start_date,end_date)
     #printing axes label 

print(panel_data.axes)
print(type(panel_data))

panel_data.info()
panel_data.describe
panel_data.columns.to_list()
#stock open and close data

stock_close=panel_data['Close']

stock_open=panel_data['Open']
stock_close.iloc[340]
print(stock_close.shape)

stock_close=np.array(stock_close).T

print(stock_close.shape)
print(stock_open.shape)

stock_open=np.array(stock_open).T

print(stock_open.shape)
row,column=stock_open.shape

row
print([name for name,id in companies],end=' ')
print(len(companies))
#we are now calculating how far the stocks have been changed during this tenure.

#defining a placeholder matrix with the dimension of the final matrix 

# movement value will be basically the sum of intraday change for the stock for the tenure considered which is 01-01-2016 till 31-12-2018 in our case



movements=np.zeros(([row,column]))

for i in range (row):

    movements[i,:]=np.subtract(stock_close[i,:],stock_open[i,:])

for i in range(len(companies)):

    print('company: {}, change:{}'.format(companies[i][0], sum(movements[i][:])))

print(movements.shape)



print(max(movements[4]))
#Now since we have the desired data, lets visualize the data now with matplotlib

plt.clf

plt.figure(figsize=(18,16))

ax1=plt.subplot(221)

plt.plot(movements[0][:])

plt.title(companies[0])

plt.subplot(222,sharey=ax1)

plt.plot(movements[1][:])

plt.title(companies[1])

plt.show()
from sklearn.preprocessing import Normalizer

normalizer=Normalizer()

new=normalizer.fit_transform(movements)

print(new.max())

print(new.min())

print(new.mean())


plt.figure(figsize=(18,16))

ax1=plt.subplot(221)

plt.plot(new[0][:])

plt.title(companies[0])

plt.subplot(222,sharey=ax1)

plt.plot(new[1][:])

plt.title(companies[1])

plt.show()
#Now we are going to make a pipeline and see how it works

from sklearn.pipeline import make_pipeline

from sklearn.cluster import KMeans

from sklearn.preprocessing import Normalizer
normalizer=Normalizer()



kmeans=KMeans(n_clusters=10,max_iter=1000)



pipeline=make_pipeline(normalizer,kmeans)

# now lets fit the data to the pipeline

pipeline.fit(movements)

print(kmeans.inertia_)

#Study more about kmeans.inertia_
labels=pipeline.predict(movements)

# print(len(labels))

# print(len(companies))

df=pd.DataFrame({'companies':companies,'labels':labels})

print(df.sort_values('labels'))
# Now we will apply pca and then apply kmeans on the data with reduced dimension

from sklearn.decomposition import PCA
reduced_data=PCA(n_components=2).fit_transform(new)
kmeans=KMeans(n_clusters=10)

model=kmeans.fit(reduced_data)

labels=model.predict(reduced_data)
df=pd.DataFrame({'companies':companies,'labels':labels,})

print(df.sort_values('labels'))
#now we will create a meshgrid using np.mesh 

# step size

h=0.1

#plotting the discision boundary

x_min,x_max=reduced_data[0].min()-1,reduced_data.max()+1

y_min,y_max=reduced_data[1].min()-1,reduced_data.max()+1

xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max, h))



a=np.arange(18).reshape(3,6)

b=np.arange(0,18,1)

print(a)

print(b)

print(a.ravel(order='A'))
z=kmeans.predict(np.c_[xx.ravel(),yy.ravel()])
print(xx.shape)

print(yy.shape)
#put the result into a color plot

z=z.reshape(xx.shape)
from matplotlib import cm



# define color plot



cmap=plt.cm.Paired



#plotting figure



plt.figure(figsize=(10,10))

plt.imshow(z,interpolation='nearest',extent=(xx.min(),xx.max(), yy.min(),yy.max()),cmap=cmap, aspect='auto',origin='lower')

plt.plot(reduced_data[:,0],reduced_data[:,1],'k.',markersize=5)



#plot the centroids of each cluster as  a white x



centroids=kmeans.cluster_centers_

plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=169,linewidths=3,color='w',zorder=10)

plt.title('Kmeans clustering on stock market movements (PCA-reduced data)')

plt.xlim(x_min,x_max)

plt.ylim(y_min,y_max)



plt.show()