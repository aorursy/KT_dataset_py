# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import  DecisionTreeClassifier
data_train = pd.read_csv('/kaggle/input/fires-from-space-australia-and-new-zeland/fire_archive_M6_96619.csv', skipinitialspace=True)
data_train.head()
data_train['brightness'].hist(bins=20)
data_train['brightness'].quantile([0, 0.25, .75, .9])
data_train['longitude'].quantile([0, 0.25, .75, .9])
data_train['latitude'].quantile([0, 0.25, .75, .9])
def bright_categorize(brightness):
     
    if brightness < 316.5:
        return 'low'
    elif 336.7 <= brightness <= 351.0:
        return 'High'
    else:
        return 'Extreme'

data_train['brightness'] = data_train['brightness'].fillna(data_train['brightness'].mean())
data_train['longitude'] = data_train['longitude'].fillna(data_train['longitude'].mean())
data_train['latitude'] = data_train['latitude'].fillna(data_train['latitude'].mean())
data_train['brightness_temperature'] = data_train['brightness'].apply(bright_categorize)
sns.barplot(x='brightness_temperature', y='latitude', data=data_train)
sns.barplot(x='brightness_temperature', y='longitude', data=data_train)
def area_categorize(longitude, latitude):
    
    if longitude < 122.8051765 or -30.000233 < latitude < -25.760321:
        return 'Western australia'
    elif 122.8051765 < longitude < 132.551000 or -20.917574 < latitude < 19.4914:
        return 'Northern territory'
    elif 132.551000 < longitude < 136.209152 or -31.840233 < latitude < -30.000233:
        return 'South Australia'
    elif 144.964600 < longitude < 145.612793 or -37.020100 < latitude < -31.840233:
        return 'New south wales'
    elif 142.702789 < longitude < 144.964600 or latitude < -37.020100:
        return 'Victoria'
    elif 136.209152 < longitude < 142.702789 or -25.760321 < latitude < -20.917574:
        return 'Queensland'
    else:
        return 'Unidentified'

df = pd.DataFrame(data_train)
df_train = pd.DataFrame(df, columns = ['brightness_temperature', 'daynight', 'brightness']) 
df_train['Area of Fire'] = df.apply(lambda x: area_categorize(x['longitude'], x['latitude']), axis=1)
df_train.head()
sns.barplot(x='brightness', y='Area of Fire', data=df_train)
import folium
m3 = folium.Map(location=[-38.043995, 145.264296], tiles='cartodbdark_matter', zoom_start=4)

for i in range(0,3000):
    df.loc[i, 'brightness']
    def color_producer(val):
        if val < 325 :
            return 'red'
        else:
            return 'orange'
for i in range(0,3000):
	folium.Circle(location=[df.iloc[i]['latitude'], df.iloc[i]['longitude']], radius=120*df.iloc[i]['brightness'], color=color_producer(df.iloc[i]['brightness'])).add_to(m3)
m3
sns.factorplot(y='Area of Fire', kind='count', hue='Area of Fire', data=df_train)
sns.barplot(x='daynight', y='brightness', data=df_train)
def preproccesing_data(df):
    df['brightness_temperature'] = df['brightness_temperature'].map({'low':0, 'High':1, 'Extreme':2})
    df['Area of Fire'] = df['Area of Fire'].map({'Western australia':0, 'Queensland':1, 'South Australia':2, 'New south wales':3, 'Northern territory':4, 'Victoria':5})
    df['daynight'] = df['daynight'].replace(['D'], 0)
    df['daynight'] = df['daynight'].replace(['N'], 1)
    return df_train


train_final = preproccesing_data(df_train)
train_final

X_train = train_final.drop("Area of Fire", axis=1).fillna(0)
Y_train = train_final["Area of Fire"]
X_test  = test_final.copy()
X_train.shape, Y_train.shape, X_test.shape
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_train)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_train)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_train)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_train)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd