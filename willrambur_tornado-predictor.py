#Modules import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from bq_helper import BigQueryHelper
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm, datasets, model_selection, preprocessing, metrics
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification



years =range(2008,2018)
helper = BigQueryHelper('bigquery-public-data', 'noaa_gsod')

sql = '''
SELECT
    year, mo, da, temp, min, max, prcp, stn, b.lat,b.lon,tornado_funnel_cloud, wdsp, gust, slp, dewp, thunder, hail, snow_ice_pellets, fog, rain_drizzle, sndp

FROM
    `bigquery-public-data.noaa_gsod.gsod{}` a

INNER JOIN
`bigquery-public-data.noaa_gsod.stations` b ON a.stn = b.usaf AND a.wban=b.wban

WHERE 
    b.country = 'US'
    AND b.state = 'TX' OR b.state = 'OK' OR b.state = 'LA' OR b.state = 'MISS' OR b.state = 'AL' OR b.state = 'AR' OR b.state = 'TN' OR b.state = 'MO' OR b.state='KY' OR b.state= 'IL' OR b.state = 'IN'
 '''
weather = [ helper.query_to_pandas(sql.format(i)) for i in years ]
weather = pd.concat(weather)


#run if you want to download CSV and use outside of kaggle
#weather.to_csv('weather.csv', index = False)
tornado_funnel_cloud = np.array(weather['tornado_funnel_cloud'])
hail = np.array(weather['hail'])
thunder = np.array(weather['thunder'])
snow_ice_pellets = np.array(weather['snow_ice_pellets'])
fog = np.array(weather['fog'])
rain_drizzle = np.array(weather['rain_drizzle'])

#print(len(thunder))
#print(len(weather))

#logical or function was being dumb
stormLabel = tornado_funnel_cloud+hail+thunder+snow_ice_pellets+fog+rain_drizzle #make new storm vector- combination of storms

stormLabel = [int(q) for q in stormLabel]

for q in range(len(stormLabel)):
    if stormLabel[q] > 1:
        stormLabel[q] = 1
weather['Storm']=stormLabel
allStorms=weather[weather['Storm']==1]
noStorm=weather[weather['Storm']==0]

#creating a 50/50 split
noStormSampled=noStorm.sample(allStorms.shape[0])

X=allStorms.append(noStormSampled)
Y=X.Storm
X=np.array(X.drop(columns=['tornado_funnel_cloud','year','mo','da','stn','lat','lon','thunder', 'hail', 'snow_ice_pellets', 'fog', 'rain_drizzle','Storm']))

normX = preprocessing.normalize(X) 
stanX = preprocessing.scale(X)
newX = preprocessing.scale(normX)

#k = X.shape[0]/2

tree1 = DecisionTreeClassifier()
tree1.fit(X,Y)
y_pred1 = cross_val_predict(tree1, newX, Y, cv=5)
print(metrics.classification_report(Y, y_pred1))
print(metrics.confusion_matrix(Y, y_pred1)) #TN, FP; FN, TP

clf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators = 100)
y_pred_forest = cross_val_predict(clf,normX,Y,cv=5)
print(metrics.classification_report(Y, y_pred_forest))
print(metrics.confusion_matrix(stormLabel, y_pred_forest)) #TN, FP; FN, TP

Tornados=allStorms[allStorms['tornado_funnel_cloud']=='1']
noTornados=allStorms[allStorms['tornado_funnel_cloud']=='0']
print(Tornados.shape)
print(noTornados.shape)
Tornados=allStorms[allStorms['tornado_funnel_cloud']=='1']
noTornados=allStorms[allStorms['tornado_funnel_cloud']=='0']
n=4*Tornados.shape[0]
noTornadoSampled=noTornados.sample(n)
X=Tornados.append(noTornadoSampled)
Y=X.tornado_funnel_cloud
X=np.array(X.drop(columns=['tornado_funnel_cloud','year','mo','da','stn','lat','lon','thunder', 'hail', 'snow_ice_pellets', 'fog', 'rain_drizzle','Storm']))
normX = preprocessing.normalize(X) 
stanX = preprocessing.scale(X)
newX = preprocessing.scale(normX)

tree2 = DecisionTreeClassifier()
tree2.fit(X,Y)
y_pred = cross_val_predict(tree2, newX, Y, cv=5)
print(metrics.classification_report(Y, y_pred))
print(metrics.confusion_matrix(Y, y_pred)) #TN, FP; FN, TP
print(y_pred1)
output1=tree1.predict(weather.drop(columns=['tornado_funnel_cloud','year','mo','da','stn','lat','lon','thunder', 'hail', 'snow_ice_pellets', 'fog', 'rain_drizzle','Storm']))
weather['output1']=output1
filtered=weather[weather['output1']==1]
output2=tree2.predict(filtered.drop(columns=['tornado_funnel_cloud','year','mo','da','stn','lat','lon','thunder', 'hail', 'snow_ice_pellets', 'fog', 'rain_drizzle','Storm','output1']))
filtered['output2']=output2
ones=filtered[filtered['output2']=='1']

TP=ones[ones['tornado_funnel_cloud']=='1']
FP=ones[ones['tornado_funnel_cloud']=='0']


zeros=filtered[filtered['output2']=='0']
TN=zeros[zeros['tornado_funnel_cloud']=='0']
FN=zeros[zeros['tornado_funnel_cloud']=='1']

print('TP = ',TP.shape[0])
print('TP = ',TP.shape[0])
print('FP = ',FP.shape[0])
print('TN = ',TN.shape[0])
print('FN = ',FN.shape[0])
TP.shape[0]/weather.shape[0]*100
snow=allStorms[allStorms['snow_ice_pellets']=='1']
noSnow=allStorms[allStorms['snow_ice_pellets']=='0']
n=4*snow.shape[0]
noSnowSampled=noSnow.sample(n)
X=snow.append(noSnowSampled)
Y=X.snow_ice_pellets
X=np.array(X.drop(columns=['tornado_funnel_cloud','year','mo','da','stn','lat','lon','thunder', 'hail', 'snow_ice_pellets', 'fog', 'rain_drizzle','Storm']))

normX = preprocessing.normalize(X) 
stanX = preprocessing.scale(X)
newX = preprocessing.scale(normX)

tree3 = DecisionTreeClassifier()
tree3.fit(X,Y)
y_pred = cross_val_predict(tree3, newX, Y, cv=5)
print(metrics.classification_report(Y, y_pred))
print(metrics.confusion_matrix(Y, y_pred)) #TN, FP; FN, TP
Tornados=allStorms[allStorms['tornado_funnel_cloud']=='1']
noTornados=allStorms[allStorms['tornado_funnel_cloud']=='0']
n=Tornados.shape[0]
noTornadoSampled=noTornados.sample(n)
X=Tornados.append(noTornadoSampled)
Y=X.tornado_funnel_cloud
X=np.array(X.drop(columns=['tornado_funnel_cloud','year','mo','da','stn','lat','lon','thunder', 'hail', 'snow_ice_pellets', 'fog', 'rain_drizzle','Storm']))

normX = preprocessing.normalize(X) 
stanX = preprocessing.scale(X)
newX = preprocessing.scale(normX)

tree3 = DecisionTreeClassifier()
tree3.fit(X,Y)
y_pred = cross_val_predict(tree3, newX, Y, cv=5)
print(metrics.classification_report(Y, y_pred))
print(metrics.confusion_matrix(Y, y_pred)) #TN, FP; FN, TP
years =range(2005,2008)
weather2 = [ helper.query_to_pandas(sql.format(i)) for i in years ]
weather2 = pd.concat(weather2)
tornado_funnel_cloud = np.array(weather2['tornado_funnel_cloud'])
hail = np.array(weather2['hail'])
thunder = np.array(weather2['thunder'])
snow_ice_pellets = np.array(weather2['snow_ice_pellets'])
fog = np.array(weather2['fog'])
rain_drizzle = np.array(weather2['rain_drizzle'])


#logical or function was being dumb
stormLabel = tornado_funnel_cloud+hail+thunder+snow_ice_pellets+fog+rain_drizzle #make new storm vector- combination of storms

stormLabel = [int(q) for q in stormLabel]

for q in range(len(stormLabel)):
    if stormLabel[q] > 1:
        stormLabel[q] = 1
weather2['Storm']=stormLabel

output1=tree1.predict(weather2.drop(columns=['tornado_funnel_cloud','year','mo','da','stn','lat','lon','thunder', 'hail', 'snow_ice_pellets', 'fog', 'rain_drizzle','Storm']))
weather2['output1']=output1
filtered=weather2[weather2['output1']==1]

output2=tree2.predict(filtered.drop(columns=['tornado_funnel_cloud','year','mo','da','stn','lat','lon','thunder', 'hail', 'snow_ice_pellets', 'fog', 'rain_drizzle','Storm','output1']))
filtered['output2']=output2

ones=filtered[filtered['output2']=='1']

TP=ones[ones['tornado_funnel_cloud']=='1']
FP=ones[ones['tornado_funnel_cloud']=='0']


zeros=filtered[filtered['output2']=='0']
TN=zeros[zeros['tornado_funnel_cloud']=='0']
FN=zeros[zeros['tornado_funnel_cloud']=='1']


print('TP = ',TP.shape[0])
print('FP = ',FP.shape[0])
print('TN = ',weather2.shape[0]-filtered.shape[0]+TN.shape[0])
print('FN = ',FN.shape[0])
