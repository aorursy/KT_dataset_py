import pandas as pd

data = pd.read_csv("../input/retail-grocery-store-sales-data-from-20162019/fruithut_data_ordered_csv_file_1_1.csv")
data.head()
data.shape
storedata = data[-200:]
storedata.shape
storedata.head()
storedata.apply(pd.Series.nunique)
storedata.drop(['TICKET', 'REFERENCE', 'CODE', 'TRANSID'], inplace=True, axis=1)
storedata.head()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(7,7))
sns.set_context("poster")
sns.scatterplot(data=storedata, x='UNITS', y='PRICE', hue='PAYMENT')
plt.figure(figsize=(7,7))
sns.scatterplot(data=storedata, x="UNITS", y="TOTAL", hue="PAYMENT")
from sklearn.preprocessing import StandardScaler, LabelEncoder

encdata = storedata.copy()
encdata.columns
lab = LabelEncoder()
for i in ['NAME', 'PAYMENT', 'CATERGORY']:
    encdata[i] = lab.fit_transform(encdata[i])
encdata['DATENEW'] = pd.to_datetime(encdata.DATENEW)
encdata['hour'] = encdata['DATENEW'].dt.hour

encdata['day'] = encdata['DATENEW'].dt.day

encdata['month'] = encdata['DATENEW'].dt.month
encdata.drop(['DATENEW'], inplace=True, axis=1)
scaled = encdata.copy()

scale = StandardScaler()

scaledata = scale.fit_transform(scaled)
from sklearn.cluster import KMeans

kms = KMeans(n_clusters=3, init="k-means++")

cluster = pd.DataFrame(kms.fit_predict(scaledata), columns = ['cluster'])
enc = encdata.reset_index()
df = pd.concat([enc, cluster], axis=1)
plt.figure(figsize=(7,7))
sns.scatterplot(data=df,x="UNITS",y="PRICE", hue='cluster')
for i, word in enumerate(df['cluster']):
    if word == 0:
        df['cluster'][i] = 'Small-Scale Buyer'
        
for i, word in enumerate(df['cluster']):
    if word == 1:
        df['cluster'][i] = 'Mid-Scale Buyer'
        
for i, word in enumerate(df['cluster']):
    if word == 2:
        df['cluster'][i] = 'Large-Size Buyer'
plt.figure(figsize=(7,7))
sns.scatterplot(data=df,x="UNITS",y="PRICE", hue='cluster')