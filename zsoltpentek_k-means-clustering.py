import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv('../input/college-data/College_Data', index_col=0)
data.head()
data.info()
data.describe()
sns.set_style('whitegrid')
plt.figure(figsize=(6,6))
sns.scatterplot(x='Room.Board', y='Grad.Rate', data=data, hue='Private', alpha=0.5)
sns.set_style('whitegrid')
plt.figure(figsize=(6,6))
sns.scatterplot(x='Outstate', y='F.Undergrad', data=data, hue='Private', alpha=0.5)
data[data['Private']=='Yes']['Outstate'].hist(color='green', alpha=0.5, figsize=(15,9), bins=18)
data[data['Private']=='No']['Outstate'].hist(color='red', alpha=0.5, bins=18)
data[data['Private']=='Yes']['Grad.Rate'].hist(color='green', alpha=0.5, figsize=(15,9), bins=18, label='private')
data[data['Private']=='No']['Grad.Rate'].hist(color='red', alpha=0.5, bins=18, label='not private')
plt.legend()
data['Grad.Rate']['Cazenovia College']=100
data[data['Private']=='Yes']['Grad.Rate'].hist(color='green', alpha=0.5, figsize=(15,9), bins=18, label='private')
data[data['Private']=='No']['Grad.Rate'].hist(color='red', alpha=0.5, bins=18, label='not private')
plt.legend()
from sklearn.cluster import KMeans
k_model = KMeans(n_clusters=2)
k_model.fit(data.drop('Private', axis=1))
k_model.cluster_centers_
def converter(private):
    if private=='Yes':
        return 1
    else:
        return 0
data['Cluster'] = data['Private'].apply(converter)
data.head()
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(data['Cluster'], k_model.labels_))
print(classification_report(data['Cluster'], k_model.labels_))