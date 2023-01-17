# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
import seaborn as sns
# df_cc = pd.read_csv('C:/Users/Vikash/Documents/Python Scripts/Data_Mining/CC GENERAL.csv')
df_cc = pd.read_csv('/kaggle/input/creditcarddata/CC GENERAL.csv')
df_cc.head(10)
df_cc.describe()
df_cc.isnull().sum().sort_values(ascending=False).head(4)

# Replace the missing values with mean value
df_cc.loc[(df_cc['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS']=df_cc['MINIMUM_PAYMENTS'].mean()
df_cc.loc[(df_cc['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT']=df_cc['CREDIT_LIMIT'].mean()
df_cc_all = df_cc[:]

columns=['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 
         'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']

for c in columns:
    Range=c + '_RANGE'
    df_cc[Range]=0        
    df_cc.loc[((df_cc[c]>0)&(df_cc[c]<=500)),Range]=1
    df_cc.loc[((df_cc[c]>500)&(df_cc[c]<=1000)),Range]=2
    df_cc.loc[((df_cc[c]>1000)&(df_cc[c]<=3000)),Range]=3
    df_cc.loc[((df_cc[c]>3000)&(df_cc[c]<=5000)),Range]=4
    df_cc.loc[((df_cc[c]>5000)&(df_cc[c]<=10000)),Range]=5
    df_cc.loc[(df_cc[c]>10000),Range]=6


df_cc.head()
columns=['PURCHASES_TRX', 'CASH_ADVANCE_TRX']  

for c in columns:    
    Range=c+'_RANGE'
    df_cc[Range]=0
    df_cc.loc[((df_cc[c]>0)&(df_cc[c]<=5)),Range]=1
    df_cc.loc[((df_cc[c]>5)&(df_cc[c]<=10)),Range]=2
    df_cc.loc[((df_cc[c]>10)&(df_cc[c]<=15)),Range]=3
    df_cc.loc[((df_cc[c]>15)&(df_cc[c]<=20)),Range]=4
    df_cc.loc[((df_cc[c]>20)&(df_cc[c]<=30)),Range]=5
    df_cc.loc[((df_cc[c]>30)&(df_cc[c]<=50)),Range]=6
    df_cc.loc[((df_cc[c]>50)&(df_cc[c]<=100)),Range]=7
    df_cc.loc[(df_cc[c]>100),Range]=8
df_cc.head()
columns=['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 
         'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT']

for c in columns:
    
    Range=c+'_RANGE'
    df_cc[Range]=0
    df_cc.loc[((df_cc[c]>0)&(df_cc[c]<=0.1)),Range]=1
    df_cc.loc[((df_cc[c]>0.1)&(df_cc[c]<=0.2)),Range]=2
    df_cc.loc[((df_cc[c]>0.2)&(df_cc[c]<=0.3)),Range]=3
    df_cc.loc[((df_cc[c]>0.3)&(df_cc[c]<=0.4)),Range]=4
    df_cc.loc[((df_cc[c]>0.4)&(df_cc[c]<=0.5)),Range]=5
    df_cc.loc[((df_cc[c]>0.5)&(df_cc[c]<=0.6)),Range]=6
    df_cc.loc[((df_cc[c]>0.6)&(df_cc[c]<=0.7)),Range]=7
    df_cc.loc[((df_cc[c]>0.7)&(df_cc[c]<=0.8)),Range]=8
    df_cc.loc[((df_cc[c]>0.8)&(df_cc[c]<=0.9)),Range]=9
    df_cc.loc[((df_cc[c]>0.9)&(df_cc[c]<=1.0)),Range]=10
df_cc.drop(['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY',  'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT' ], axis=1, inplace=True)

X= np.asarray(df_cc)
scale = StandardScaler()
X = scale.fit_transform(X)
X.shape

cluster_number = 15 
kmean_sq_value =[]
for i in range(1, cluster_number):
    kmean = KMeans(i)
    kmean.fit(X)
    kmean_sq_value.append([i, kmean.inertia_])

kmean_sq_value_df = pd.DataFrame(kmean_sq_value)
plt.plot(kmean_sq_value_df[0], kmean_sq_value_df[1], 'g*-') 
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors')
plt.show()

# Let's choose n=7 as neither we want to have many cluster nor we see much improvement
# Adding the clusters
n_clusters = 7
kmean = KMeans(n_clusters).fit(X)
labels = kmean.labels_
# Convert labels to dataframe
df_labels = pd.DataFrame(labels)
df_labels.rename(columns={0: 'cluster'}, inplace=True)
df_clusters = df_cc.merge(df_labels, how="inner", left_index=True, right_index=True)
for c in df_clusters:
    grid= sns.FacetGrid(df_clusters, col='cluster')
    grid.map(plt.hist, c)
#Normalize data
df_normalized = normalize(X)

#Reducing the dimension
pca = PCA(2)
X_PCA = pca.fit_transform(df_normalized)
X_PCA = pd.DataFrame(X_PCA)
X_PCA.columns = ['C1', 'C2']
X_PCA.head()

# Visualizing the cluster
cluster_groups = [
    [0, "people with high credit limit"],
    [1, "who make all type of purchases"], 
    [2, "people with due payments"],         
    [3, "purchases mostly in installments"],         
    [4, "who take more cash in advance"],          
    [5, "who make expensive purchases"],         
    [6, "who don't spend much money"]]

dfc = pd.DataFrame(KMeans(n_clusters).fit_predict(X_PCA))
dfc = dfc.rename(columns={0:'cluster_group'})
X_df = pd.concat([X_PCA, dfc], axis=1)

plt.subplots(figsize=(10, 6))
plt.scatter(X_df['C1'], X_df['C2'], c=X_df['cluster_group'],
cmap = plt.cm.get_cmap('plasma', 7))
plt.title("Customers Credit Card usage behaviour")

cbar = plt.colorbar()
plt.clim(-0.5, 6.5)
cbar.ax.set_yticklabels(x[1] for x in cluster_groups)
plt.axis("tight") 
plt.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
plt.show()
df_cc_limit_payments = df_cc_all[['CREDIT_LIMIT', 'PAYMENTS']]

plt.figure(figsize=(10, 6))
ax = sns.regplot(x='CREDIT_LIMIT', y='PAYMENTS', data=df_cc_limit_payments, color='green', marker='.', scatter_kws={'s': 200})
ax.set(xlabel='CREDIT_LIMIT', ylabel='PAYMENTS') 
ax.set_title('Credit Limit vs Payments') 
plt.grid(True)
y_column = 'PURCHASES'
x_column = 'CREDIT_LIMIT'
columns = [x_column, y_column]
df_cc_limit_payments = df_cc_all[columns]

plt.figure(figsize=(10, 6))
ax = sns.regplot(x=x_column, y=y_column, data=df_cc_limit_payments, color='green', marker='.', scatter_kws={'s': 200})
ax.set(xlabel=x_column, ylabel=y_column) 
ax.set_title( x_column + ' vs ' + y_column) 
plt.grid(True)
y_column = 'PURCHASES'
x_column = 'CREDIT_LIMIT'
columns = [x_column, y_column]
df_cc_limit_payments = df_cc_all[columns]
df_cc_limit_payments = df_cc_limit_payments[df_cc_limit_payments['CREDIT_LIMIT']>=16000]

plt.figure(figsize=(10, 6))
ax = sns.regplot(x=x_column, y=y_column, data=df_cc_limit_payments, color='green', marker='.', scatter_kws={'s': 200})
ax.set(xlabel=x_column, ylabel=y_column) 
ax.set_title( x_column + ' vs ' + y_column) 
plt.grid(True)
x_column = 'CASH_ADVANCE'
y_column = 'PAYMENTS'
columns = [x_column, y_column]
df_cc_limit_payments = df_cc_all[columns]
df_cc_limit_payments = df_cc_limit_payments[df_cc_limit_payments['CASH_ADVANCE']>=10000]
plt.figure(figsize=(10, 6))
ax = sns.regplot(x=x_column, y=y_column, data=df_cc_limit_payments, color='green', marker='.', scatter_kws={'s': 200})

ax.set(xlabel=x_column, ylabel=y_column) 
ax.set_title( x_column + ' vs ' + y_column) 
plt.grid(True)

