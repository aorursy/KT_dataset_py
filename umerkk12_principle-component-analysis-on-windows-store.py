import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
df = pd.read_csv('../input/windows-store/msft.csv')
df.head()
df.info()
df.describe()
msno.matrix(df)
df.dropna(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.info()
latest_date = df['Date'].max()

df['Recency (x days ago)'] =df['Date'] - latest_date 
df['Recency (x days ago)'] = df['Recency (x days ago)'].dt.days
df.head()
df.groupby(df['Price']).count()# found out that only the value 'Free' was non-numeric
df['Price'] = df['Price'].replace('Free', '0')# Changed it to numeric
df.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df[['Rating', 'No of people Rated', 'Recency (x days ago)']])
scaled_data = scaler.transform(df[['Rating', 'No of people Rated', 'Recency (x days ago)']])
scaled_data = pd.DataFrame(scaled_data)
from sklearn.decomposition import PCA
logmodel = PCA(n_components=2)
logmodel.fit(scaled_data)
pca_x = logmodel.transform(scaled_data)
scaled_data.shape
pca_x.shape
df_comp = pd.DataFrame(logmodel.components_, columns = ['Rating', 'No of people Rated', 'Recency (x days ago)'])
plt.figure(figsize=(10,6))
sns.heatmap(df_comp)
plt.figure(figsize=(10,6))
ty=sns.scatterplot(pca_x[:,0], pca_x[:,1])
sns.despine(left=True)
ty.set_title('PCA Results')
ty.set_ylabel('Second Principle Component ')
ty.set_xlabel('First Principle Component ')



pca_x