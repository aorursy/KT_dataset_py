import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import misc

from sklearn import preprocessing

from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KDTree, BallTree, KNeighborsClassifier

from sklearn.linear_model import LogisticRegressionCV

from sklearn import svm

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import utils

%matplotlib inline



data = pd.read_csv('../input/marvel-wikia-data.csv')

data.head()
data.drop('urlslug', axis=1, inplace=True)

data.drop('EYE', axis=1, inplace=True)

data.drop('HAIR', axis=1, inplace=True)

data.drop('FIRST APPEARANCE', axis=1, inplace=True)

data.drop('name', axis=1, inplace=True)

data.head()

fig = plt.figure(figsize=(10,4))

fig.add_subplot(121)

data.ALIGN[data['SEX'] == 'Male Characters'].value_counts().plot(kind='pie')

fig.add_subplot(122)

data.ALIGN[data['SEX'] == 'Female Characters'].value_counts().plot(kind='pie')
data['ALIGN'].replace('Good Characters',2, inplace = True)

data['ALIGN'].replace('Bad Characters',1, inplace = True)

data['ALIGN'].replace('Neutral Characters',3, inplace = True)

data['SEX'].replace('Male Characters', 2, inplace = True)

data['SEX'].replace('Female Characters', 1, inplace = True)

data['SEX'].replace('Genderfluid Characters', 3, inplace = True)

data['SEX'].replace('Agender Characters', 4, inplace = True)

data['ALIVE'].replace('Living Characters', 1, inplace = True)

data['ALIVE'].replace('Deceased Characters', 0, inplace = True)

data['ID'].replace('Secret Identity', 2, inplace = True)

data['ID'].replace('Public Identity', 1, inplace = True)

data['ID'].replace('No Dual Identity', 3, inplace = True)

data['ID'].replace('Known to Authorities Identity', 4, inplace = True)

data['GSM'].replace('Bisexual Characters', 1, inplace = True)

data['GSM'].replace('Transvestites', 2, inplace = True)

data['GSM'].replace('Homosexual Characters', 3, inplace = True)

data['GSM'].replace('Pansexual Characters', 4, inplace = True)

data['GSM'].replace('Transgender Characters', 5, inplace = True)

data['GSM'].replace('Genderfluid Characters', 6, inplace = True)

data.replace(np.nan, 0,inplace=True)

data.ALIVE = data.ALIVE.astype(int)

data.head()

gsm = data['GSM'].values.reshape(-1,1)

min_max_scaler = preprocessing.MinMaxScaler()

gsm_scaled = min_max_scaler.fit_transform(gsm)

data['GSM'] = pd.DataFrame(gsm_scaled)



id = data['ID'].values.reshape(-1,1)

min_max_scaler = preprocessing.MinMaxScaler()

id_scaled = min_max_scaler.fit_transform(id)

data['ID'] = pd.DataFrame(id_scaled)





sex = data['SEX'].values.reshape(-1,1)

min_max_scaler = preprocessing.MinMaxScaler()

sex_scaled = min_max_scaler.fit_transform(sex)

data['SEX'] = pd.DataFrame(sex_scaled)



align = data['ALIGN'].values.reshape(-1,1)

min_max_scaler = preprocessing.MinMaxScaler()

align_scaled = min_max_scaler.fit_transform(align)

data['ALIGN'] = pd.DataFrame(align_scaled)



page_id = data['page_id'].values.reshape(-1,1)

min_max_scaler = preprocessing.MinMaxScaler()

page_id_scaled = min_max_scaler.fit_transform(page_id)

data['page_id'] = pd.DataFrame(page_id_scaled)



app = data['APPEARANCES'].values.reshape(-1,1)

min_max_scaler = preprocessing.MinMaxScaler()

app_scaled = min_max_scaler.fit_transform(app)

data['APPEARANCES'] = pd.DataFrame(app_scaled)



year = data['Year'].values.reshape(-1,1)

min_max_scaler = preprocessing.MinMaxScaler()

year_scaled = min_max_scaler.fit_transform(year)

data['Year'] = pd.DataFrame(year_scaled)



data.head()
data['ALIVE'].corr(data['GSM'])
#split dataset in features and target variable

# feature_cols = ['ALIVE', 'ID', 'SEX', 'APPEARANCES','year','bp','pedigree']

# X = pima[feature_cols] # Features

# y = pima.label # Target variable'ID', 'SEX', 'GSM', 'APPEARANCES']], data['ALIVE'], test_size = 0.3)



trainX, testX, trainY, testY = train_test_split(data[['GSM', 'SEX', 'APPEARANCES','Year']], data['ALIVE'], test_size = 0.3)
clf1 = DecisionTreeClassifier()

clf1.fit(trainX, trainY)

accuracy = clf1.score(testX, testY)

print(accuracy)
clf3 = LogisticRegressionCV()

clf3.fit(trainX, trainY)

accuracy = clf3.score(testX, testY)

print(accuracy)
clf6 = RandomForestClassifier()

clf6.fit(trainX, trainY)

accuracy = clf6.score(testX, testY)

print(accuracy)
clf4 = KNeighborsClassifier()

clf4.fit(trainX, trainY)

accuracy = clf4.score(testX, testY)

print(accuracy)
clf5 = MLPClassifier()

clf5.fit(trainX, trainY)

accuracy = clf5.score(testX, testY)

print(accuracy)
data.head()