import pandas as pd
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

datafile = '../input/air_data.csv'
data = pd.read_csv(datafile, encoding = 'utf-8')
data.head()
explore = data.describe(percentiles = [], include = 'all').T 
# Obtain the number of null record
explore['null'] = len(data) - explore['count'] 
# Return the number of missing value, min and max
explore = explore[['null','min','max']] 
explore
# Keep instances that ticket price is not null 
clean = data[data['SUM_YR_1'].notnull() & data['SUM_YR_2'].notnull()] 
# Keep instances that ticket price is not 0  OR  average discount is not 0 (100% off)
index1 = clean['SUM_YR_1'] != 0
index2 = clean['SUM_YR_2'] != 0
index3 = clean['avg_discount'] != 0
clean = clean[index1 | index2 | index3] 
clean.head()
reduce = clean[['LOAD_TIME', 'FFP_DATE', 'LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount']]
reduce.head()
pd.options.mode.chained_assignment = None  # default='warn'
reduce['LOAD_TIME'] = pd.to_datetime(reduce['LOAD_TIME'])
reduce['FFP_DATE'] = pd.to_datetime(reduce['FFP_DATE'])
reduce['L'] = reduce['LOAD_TIME'] - reduce['FFP_DATE']
reduce['L'] = reduce['L'].astype(dt.timedelta).map(lambda x: np.nan if pd.isnull(x) else x.days)
air_data = reduce[['L', 'LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount']]
air_data.columns = ['L','R','F','M','C']
air_data.head()
air_data_summary = air_data.describe(percentiles = [], include = 'all')
air_data_summary
air_data = (air_data - air_data.mean(axis = 0)) / (air_data.std(axis = 0))
air_data.head()
# # Elbow method to check the best fit K between 2-10
# X = air_data
# distorsions = []
# for k in range(2, 8):
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(X)
#     distorsions.append(kmeans.inertia_)

# plt.plot(range(2, 8), distorsions)
# plt.grid(True)
# plt.xlabel('K')
# plt.ylabel('Distorsions')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()
kmeans = KMeans(n_clusters= 5, n_jobs= 4)
kmeans.fit(air_data)
df = pd.DataFrame(kmeans.cluster_centers_,
               index=[0,1,2,3,4],
               columns=['L','R','F','M','C'])
df['count'] = pd.Series(kmeans.labels_).value_counts()  # Number of each clusters
df
x=[1,2,3,4,5] # match to L R F M C 
colors=['green','red','yellow','blue','black']
for i in range(5):
    plt.plot(x,kmeans.cluster_centers_[i],label=('customer%d'%(i)),linewidth=2,color=colors[i],marker='o')
    plt.legend()
plt.xlabel('L R F M C')
plt.ylabel('values')
plt.show()













































