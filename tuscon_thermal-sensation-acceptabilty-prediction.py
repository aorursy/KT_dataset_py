# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn import metrics

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.basemap import Basemap

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/ashrae-global-thermal-comfort-database-ii/ashrae_db2.01.csv',index_col='Data contributor')

data.head()
data.describe()
data.drop(columns=['Publication (Citation)'],inplace=True)
data['Database'].unique()
data['Year'].fillna(2002,inplace=True)
t = np.mean(data['Age'])

data['Age'].fillna(t,inplace=True)
data.replace(['Summer', 'Autumn', 'Winter', 'Spring', 'nan'], [1, 2, 3,4,5], inplace=True)



data['Season'].fillna(2,inplace=True)
data['Thermal sensation'].isnull().sum()

data['Thermal sensation'].unique()

a = data['Thermal sensation'].mean()

data['Thermal sensation'].fillna(a,inplace=True)
data['Thermal sensation acceptability'].isnull().sum()

data['Thermal sensation acceptability'].fillna(1,inplace=True)
data['Air movement acceptability'].unique()

data['Air movement acceptability'].isnull().sum()

data['Air movement acceptability'].fillna(1,inplace=True)
data['Koppen climate classification'].unique()

climate = {'Cfa':1, 'Csb':2, 'Aw':3, 'BSh':4, 'Csa':5, 'Csc':6, 'Dwa':7, 'Af':8, 'Cfb':9, 'BWh':10,

       'BSk':11, 'Am':12, 'As':13, 'Cwa':14, 'Cwb':15, 'Dfb':16}

data['Koppen climate classification'] = data['Koppen climate classification'].map(climate)
col_conversion = LabelEncoder()

# data['Climate'].unique()

data['Climate'] = col_conversion.fit_transform(data['Climate'])

category = col_conversion.classes_

print("Category of Columns : ",category)
data['Country'].unique()

g = LabelEncoder()

data['Country'] = g.fit_transform(data['Country'])
data['Cooling startegy_building level'].unique()

data['Cooling startegy_building level'].fillna(1,inplace = True)

data['Cooling startegy_building level'].isnull().sum()
data['Cooling startegy_building level'].unique()

data.replace([1,'Air Conditioned', 'Naturally Ventilated', 'Mixed Mode', 

       'Mechanically Ventilated'], [1, 2, 3,4,5], inplace=True)
data['City'].fillna(0,inplace = True)

data['City'].unique()

data.replace(['Tokyo', 'Texas', 'Berkeley', 'Chennai', 'Hyderabad', 'Ilam',

       'San Francisco', 'Alameda', 'Philadelphia', 'Guangzhou',

       'Changsha', 'Yueyang', 'Harbin', 'Beijing', 'Chaozhou', 'Nanyang',

       'Makati', 'Sydney', 'Jaipur', 'Kota Kinabalu', 'Kuala Lumpur', 0,

       'Beverly Hills', 'Putra Jaya', 'Kinarut', 'Kuching', 'Bedong',

       'Bratislava', 'Elsinore', 'Gabes', 'Gafsa', 'El Kef', 'Sfax',

       'Tunis', 'Midland', 'London', 'Lyon', 'Gothenburg', 'Malmo',

       'Porto', 'Halmstad', 'Athens', 'Lisbon', 'Florianopolis',

       'BrasÌ_lia', 'Recife', 'Maceio', 'Seoul', 'Tsukuba', 'Lodi',

       'Varese', 'Imola', 'Shanghai', 'Liege', 'Mexicali', 'Hermosillo',

       'Colima', 'Culiacan ', 'MÌ©rida', 'Tezpur', 'Imphal', 'Shilong',

       'Ahmedabad', 'Bangalore', 'Delhi', 'Shimla', 'Bandar Abbas',

       'Karlsruhe', 'Bauchi', 'Stuttgart', 'Hampshire', 'Wollongong',

       'Goulburn', 'Singapore', 'Cardiff', 'Bangkok', 'Jakarta',

       'Montreal', 'Brisbane', 'Darwin', 'Melbourne', 'Ottawa', 'Karachi',

       'Multan', 'Peshawar', 'Quetta', 'Saidu Sharif', 'Oxford',

       'San Ramon', 'Palo Alto', 'Walnut Creek', 'Townsville',

       'Liverpool', 'St Helens', 'Chester', 'Grand Rapids', 'Auburn',

       'Kalgoorlie', 'Honolulu'], [1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,

                                   48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,

                                   87,88,89,90,91,92,93,94,95,96,97,98,99], inplace=True)
data['Building type'].unique()

data['Building type'].fillna(6, inplace= True) 

data.replace(['Classroom', 'Office', 'Others', 'Multifamily housing',

       'Senior center',6], [1, 2, 3,4,5,6], inplace=True)

data['Cooling startegy_operation mode for MM buildings'].unique()

data['Cooling startegy_operation mode for MM buildings'].fillna(1,inplace = True)

data.replace([1,2,3,'Unknown'],[1,2,3,4],inplace=True)
# data.drop(columns=['Heating strategy_building level'],axis=1)

data['Heating strategy_building level'].unique()

data['Heating strategy_building level'].fillna(0,inplace=True)

data.replace([0,'Mechanical Heating'],[0,1],inplace=True)
data['Sex'].unique()

data['Sex'].isnull().sum()

data['Sex'].fillna(0,inplace=True)

data.replace([0,'Female','Male'],[0,0,1],inplace=True)

data['Thermal preference'].unique()

data['Thermal preference'].isnull().sum()

data['Thermal preference'].fillna(0,inplace=True)

data.replace(['warmer', 'no change', 'cooler', 0],[1,2,3,0],inplace=True)

data['Air movement preference'].unique()

data['Air movement preference'].fillna(0,inplace=True)

data.replace([2, 'more', 'less', 'nan'],[2,4,3,1],inplace=True)

data['Air movement preference'].shape

data['Air movement preference'].isnull().sum()

data['PMV'].isnull().sum()

data['PMV'].mean()

data['PMV'].fillna(1.0862311565319103,inplace=True)



data['PPD'].isnull().sum()

data['PPD'].mean()

data['PPD'].fillna(20.962094284773425,inplace=True)



data['SET'].isnull().sum()

data['SET'].mean()

data['SET'].fillna(25.769629,inplace=True)



data['Clo'].isnull().sum()

data['Clo'].mean()

data['Clo'].fillna(0.675876,inplace=True)



data['Met'].isnull().sum()

data['Met'].mean()

data['Met'].fillna(1.206626,inplace=True)
data.drop(columns=['Subject«s height (cm)','Subject«s weight (kg)','Blind (curtain)','Fan','Window','Door','Heater'],axis=1,inplace=True)

data.drop(['Database'],axis=1,inplace=True)

data['activity_10'].isnull().sum()

data['activity_10'].mean()

data['activity_10'].fillna(1.194218,inplace=True)



data['activity_20'].isnull().sum()

data['activity_20'].mean()

data['activity_20'].fillna(1.257274,inplace=True)



data['activity_30'].isnull().sum()

data['activity_30'].mean()

data['activity_30'].fillna(1.264003,inplace=True)



data['activity_60'].isnull().sum()

data['activity_60'].mean()

data['activity_60'].fillna(1.319214,inplace=True)



data['Air temperature (C)'].isnull().sum()

data['Air temperature (C)'].mean()

data['Air temperature (C)'].fillna(24.496358,inplace=True)



data['Air temperature (F)'].isnull().sum()

data['Air temperature (F)'].mean()

data['Air temperature (F)'].fillna(76.090540	,inplace=True)



data['Ta_h (C)'].isnull().sum()

data['Ta_h (C)'].mean()

data['Ta_h (C)'].fillna(24.569258,inplace=True)



data['Ta_h (F)'].isnull().sum()

data['Ta_h (F)'].mean()

data['Ta_h (F)'].fillna(76.223719	,inplace=True)



data['Ta_m (C)'].isnull().sum()

data['Ta_m (C)'].mean()

data['Ta_m (C)'].fillna(24.220964,inplace=True)



data['Ta_l (C)'].isnull().sum()

data['Ta_l (C)'].mean()

data['Ta_l (C)'].fillna(23.450124,inplace=True)



data['Ta_l (F)'].isnull().sum()

data['Ta_l (F)'].mean()

data['Ta_l (F)'].fillna(74.207647,inplace=True)



data['Operative temperature (C)'].isnull().sum()

data['Operative temperature (C)'].mean()

data['Operative temperature (C)'].fillna(24.504233,inplace=True)



data['Operative temperature (F)'].isnull().sum()

data['Operative temperature (F)'].mean()

data['Operative temperature (F)'].fillna(76.105627,inplace=True)



data['Radiant temperature (C)'].isnull().sum()

data['Radiant temperature (C)'].mean()

data['Radiant temperature (C)'].fillna(24.602735,inplace=True)



data['Radiant temperature (F)'].isnull().sum()

data['Radiant temperature (F)'].mean()

data['Radiant temperature (F)'].fillna(76.283592	,inplace=True)



data['Globe temperature (C)'].isnull().sum()

data['Globe temperature (C)'].mean()

data['Globe temperature (C)'].fillna(24.621170,inplace=True)



data['Globe temperature (F)'].isnull().sum()

data['Globe temperature (F)'].mean()

data['Globe temperature (F)'].fillna(76.316978	,inplace=True)



data.drop(columns=["Velocity_h (m/s)", "Velocity_h (fpm)",	"Velocity_m (m/s)",	"Velocity_m (fpm)",	"Velocity_l (m/s)",	"Velocity_l (fpm)"], axis=1, inplace=True)
data['Tg_h (C)'].isnull().sum()

data['Tg_h (C)'].mean()

data['Tg_h (C)'].fillna(24.796730,inplace=True)



data['Tg_h (F)'].isnull().sum()

data['Tg_h (F)'].mean()

data['Tg_h (F)'].fillna(76.631297,inplace=True)



data['Tg_m (C)'].isnull().sum()

data['Tg_m (C)'].mean()

data['Tg_m (C)'].fillna(24.375786,inplace=True)



data['Tg_m (F)'].isnull().sum()

data['Tg_m (F)'].mean()

data['Tg_m (F)'].fillna(75.874689,inplace=True)



data['Tg_l (C)'].isnull().sum()

data['Tg_l (C)'].mean()

data['Tg_l (C)'].fillna(22.970135,inplace=True)



data['Tg_l (F)'].isnull().sum()

data['Tg_l (F)'].mean()

data['Tg_l (F)'].fillna(73.341419,inplace=True)

data['Relative humidity (%)'].isnull().sum()

data['Relative humidity (%)'].mean()

data['Relative humidity (%)'].fillna(47.548293,inplace=True)



data['Humidity sensation'].isnull().sum()

data['Humidity sensation'].mean()

data['Humidity sensation'].fillna(11.470175,inplace=True)



data['Air velocity (m/s)'].isnull().sum()

data['Air velocity (m/s)'].mean()

data['Air velocity (m/s)'].fillna(0.847680,inplace=True)



data['Air velocity (fpm)'].isnull().sum()

data['Air velocity (fpm)'].mean()

data['Air velocity (fpm)'].fillna(34.932351,inplace=True)



data['Outdoor monthly air temperature (C)'].isnull().sum()

data['Outdoor monthly air temperature (C)'].mean()

data['Outdoor monthly air temperature (C)'].fillna(17.446746,inplace=True)



data['Outdoor monthly air temperature (F)'].isnull().sum()

data['Outdoor monthly air temperature (F)'].mean()

data['Outdoor monthly air temperature (F)'].fillna(63.383538,inplace=True)

data['Humidity preference'].unique()

data['Humidity preference'].fillna(0,inplace=True)

data.replace([0, 'drier', 2, 'more humid'],[0,1,2,3],inplace=True)

data.info()
visuals = pd.read_csv('../input/ashrae-global-thermal-comfort-database-ii/ashrae_db2.01.csv')



visuals['City'].fillna('Indore',inplace=True)



# Set the dimension of the figure

my_dpi=96

plt.figure(figsize=(2600/my_dpi, 1800/my_dpi), dpi=my_dpi)

 



# Make the background map

m=Basemap(llcrnrlon=-180, llcrnrlat=-65,urcrnrlon=180,urcrnrlat=80)

m.drawmapboundary(fill_color='black', linewidth=0)

m.fillcontinents(color='grey', alpha=0.3)

m.drawcoastlines(linewidth=0.1, color="red")

 

# Add a point per position

m.scatter(visuals['Country'], visuals['City'], alpha=0.4, cmap="twilight_shifted_r")

 

# copyright and source data info

plt.text( -170, -58,'Created By : Rudresh Joshi', ha='left', va='bottom', size=12, color='red' )

 

# Save as png

# plt.savefig('#315_Tweet_Surf_Bubble_map1.png', bbox_inches='tight')



X = data.drop(columns=['Ta_m (F)','Thermal comfort','Thermal sensation acceptability'])

y = data[['Thermal sensation acceptability']].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 12)
from keras.models import Sequential

from keras.layers import *
model = Sequential()

model.add(Dense(100,input_dim=51,activation="relu"))

model.add(Dense(200,activation="relu"))

model.add(Dense(100,activation="relu"))

model.add(Dense(1,activation="softmax"))

model.summary()
model.compile(loss="categorical_crossentropy",optimizer="adam", metrics=['accuracy'])
history = model.fit(

    X_train,

    y_train,

    epochs=5,

    shuffle=True,

    verbose=2,

    batch_size = 5



)
Prediction = model.predict(X_test)

Prediction = pd.DataFrame(Prediction)

Prediction
print("Accuracy:",metrics.accuracy_score(y_test, Prediction))

print("Precision:",metrics.precision_score(y_test, Prediction))

print("Recall:",metrics.recall_score(y_test, Prediction))