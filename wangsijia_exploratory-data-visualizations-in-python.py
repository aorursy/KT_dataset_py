### Set up some logistics

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
train = pd.read_csv('../input/train.csv')
### Shape of data

train.shape
### Columns of data

train.columns
### Sample data

train.head()
### Completeness of data

missing = train.isnull().sum()

missing.sort(ascending=False)

missing[0:20]
import missingno as mn

mn.matrix(train.iloc[:,0:40])
mn.matrix(train.iloc[:,40:])
mn.heatmap(train)
mn.dendrogram(train)
categorical = train.select_dtypes(include=['object']).columns.values

numeric = train.select_dtypes(include=['float64', 'int64']).columns.values
sns.countplot('LotShape',data=train)
sns.boxplot('LandContour','SalePrice',data=train)
sns.violinplot('HouseStyle','SalePrice',data=train)
sns.factorplot('RoofStyle','SalePrice',data=train)
sns.kdeplot(train['SalePrice'],shade=True)
sns.regplot('LotFrontage','SalePrice',data=train)
sns.jointplot('LotFrontage','SalePrice',kind='reg',data=train)
Categorical_Variable = train[categorical]

fig,ax = plt.subplots(17,3,figsize=(30,90),sharey='row')

fig.tight_layout()

for i in range(17):

    for j in range(3):

        if((j+i*3)<len(Categorical_Variable.columns)):

            sns.boxplot(Categorical_Variable.iloc[:,(i*3+j)],train['SalePrice'],ax=ax[i, j])
### Numeric Variable Correlations

plt.figure(figsize=(15, 15))

sns.heatmap(train[numeric].corr(), vmax=1, square=True)
### Relationship among a group of variables

sns.pairplot(train[['GarageCars', 'GarageArea','CentralAir','YearBuilt']],hue='CentralAir',dropna=True)
#Get Latitude and Longitude information from Google Maps

Neighborhood = pd.Series(train['Neighborhood'].unique())

Latitute = pd.Series([42.02430750000001,42.04163980000001,42.028025,42.0483583,

            41.9903084,42.05262800000001,42.04225588928407,42.028662,

            42.02894990000001,42.0264466,42.046226,42.05602272533084,

            42.0339027,42.02158370000001,42.0087108,42.0153961,

            42.0005053,42.1069288,42.0594717,42.03609670000001,

            41.903737,42.05641859999999,42.52281079999999,42.02279876924051,42.02528542979918])

Longitute = pd.Series([-93.63776339999998,-93.64912199999998,-93.60713989999999,-93.64671069999997,

             -93.60105329999999,-93.64458200000001,-93.67101289099082,-93.61730299999999,

             -93.62974759999997,-93.66832699999998,-93.6529104,-93.64080048864707,

             -93.67706579999998,-93.6687523,-93.6749451,-93.6853572,

             -93.64974869999998,-93.64966279999999,-93.63333649999998,-93.6488301,

             -93.603069,-93.6352364,-93.28389270000002,-93.654121402069,-93.66673850803636])

Location_Data = pd.concat([Neighborhood,Latitute,Longitute],axis=1)

Location_Data.columns = ['Neighborhood','Latitute','Longitute']
### Get median price of the neighborhood housing price and number of houses sold

volume_median = train.groupby('Neighborhood').aggregate({'Id':'count','SalePrice':'median'})

volume_median.reset_index(inplace=True)

Location_Data2= pd.merge(Location_Data,volume_median,on='Neighborhood',how='left')

Location_Data2.columns = ['Neighborhood','Latitute','Longitute','Count','SalePrice']

Location_Data2.head()
from mpl_toolkits.basemap import Basemap

lat = Location_Data2['Latitute'].values

lon = Location_Data2['Longitute'].values

volume = Location_Data2['Count'].values

price = Location_Data2['SalePrice'].values
fig = plt.figure(figsize=(8,8))

m = Basemap(llcrnrlon=-93.7,llcrnrlat=41.95,urcrnrlon=-93.55,urcrnrlat=42.08,epsg=4269)

m.shadedrelief()

m.drawstates(color='grey')

m.scatter(lon,lat,latlon=True,c=price,s=volume,cmap='Reds',alpha=0.5)

m.arcgisimage(service='World_Physical_Map',xpixels=5000,verbose=True)

### Download shp files from online resources. Coulld not find any file related to Ames' neighborhood so

### the visualization is not really helpful -just serve the purpose of demonstration

#m.readshapefile('C:/Users/Insights/Desktop/Kaggle Predicting Housing Price/iowa_administrative/iowa_administrative','iowa_administrative')

#m.readshapefile('C:/Users/Insights/Desktop/Kaggle Predicting Housing Price/iowa_highway/iowa_highway','iowa_highway')

plt.colorbar()



for a in [10,50,100,200]:

    plt.scatter([],[],c='k',alpha=0.5,s=a,label=str(a))

plt.legend(scatterpoints=1,frameon=False,labelspacing=1,loc='lower left')
