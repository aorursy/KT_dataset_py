import pandas as pd

data = pd.read_csv("../input/supermarket-sales/supermarket_sales - Sheet1.csv")
data.isnull().sum()
# Convertinf the Date column into a usable format:
data['Date'] = pd.to_datetime(data.Date)
data['day'] = data['Date'].dt.day.astype('object')

data['month'] = data['Date'].dt.month.astype('object')

data['year'] = data['Date'].dt.year.astype('object')
data.apply(pd.Series.nunique)
data.drop(['Date', 'year', 'Tax 5%', 'gross margin percentage', 'Branch', 'Invoice ID'], axis=1, inplace=True)
data['Time'] = pd.to_datetime(data.Time)
data['hour'] = data['Time'].dt.hour.astype('object')
data.drop(['Time'], inplace=True, axis=1)
data.head()
# Making another smaller dataset for later use:
newdataset = data[:200]
labels = pd.Series(newdataset['Total'])
cbedata = newdataset.copy()

# Defining the categorical columns: 
catcols = ['City', 'Customer type', 'Gender', 'Product line', 'day', 'month', 'hour', 'Payment']

# Importing CatBoostEncoder
from category_encoders import CatBoostEncoder

ce = CatBoostEncoder()

# Fitting it on the data:
ce.fit(cbedata[catcols], labels)

# Transforming the data:
cbedata[catcols] = ce.transform(cbedata[catcols])
cbedata.head()
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

scaleddata = pd.DataFrame(scale.fit_transform(cbedata), columns=cbedata.columns)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(30,10))
sns.set_context('poster', font_scale=0.7)
sns.heatmap(scaleddata.corr(), annot=True)
plt.figure(figsize=(20,7))
sns.countplot(data = newdataset, x = 'day')
plt.figure(figsize=(15, 7))
sns.set_palette(['skyblue'])
sns.scatterplot(data = newdataset, x = 'gross income', y = 'Unit price')
from sklearn.cluster import KMeans

kms = KMeans(n_clusters=4, init="k-means++")

kcluster = pd.DataFrame(kms.fit_predict(cbedata), columns=['kcluster'])
plt.figure(figsize=(15,7))
sns.set_context("poster")
sns.scatterplot(data=clustereddata, x='Unit price',
                y='gross income', hue='kcluster')
richcustomers = clustereddata.loc[clustereddata['kcluster'] == 0]
notrichcustomers = clustereddata.loc[clustereddata['kcluster'] == 2]
plt.figure(figsize=(15,5))
sns.set_context("poster", font_scale=.7)
sns.countplot(richcustomers['City'])
plt.figure(figsize=(15,5))
sns.set_context("poster", font_scale=.5)
sns.countplot(richcustomers['Product line'])
plt.figure(figsize=(15,5))
sns.set_context("poster", font_scale=.5)
sns.set_palette(['red'])
sns.distplot(richcustomers['Total'])
plt.figure(figsize=(15,5))
sns.set_context("poster", font_scale=.5)
sns.distplot(notrichcustomers['Total'])
plt.xlim(right=1200)
plt.figure(figsize=(20,7))
plt
sns.countplot(richcustomers['day'])
plt.figure(figsize=(20,7))
sns.countplot(notrichcustomers['day'])