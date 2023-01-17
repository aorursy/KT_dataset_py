import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



data = pd.read_csv("../input/TEST.csv")

data.head()
data = data.replace('?', np.NaN)



data.describe()
data.info()
for i in data.columns:

    print(i,':',data[i].unique())
#fill nan

data['race'] = data['race'].fillna(data['race'].mode()[0])

data['weight'] = data['weight'].fillna(data['weight'].mode()[0])

data['payer_code'] = data['payer_code'].fillna(data['payer_code'].mode()[0])

data['medical_specialty'] = data['medical_specialty'].fillna(data['medical_specialty'].mode()[0])

data['diag_1'] = data['diag_1'].fillna(data['diag_1'].mode()[0])

data['diag_2'] = data['diag_2'].fillna(data['diag_2'].mode()[0])

data['diag_3'] = data['diag_3'].fillna(data['diag_3'].mode()[0])
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

for col in data.columns:

    if(data[col].dtype == np.object):

        le.fit(data[col])

        data[col] = le.transform(data[col])
for i in data.columns:

    print(i,':',data[i].unique())
f, ax = plt.subplots(figsize=(20, 16))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot = True);
#since a lot of them exhibit almost zero correlation between each other or very slightly -ve corr, we can 

#remove most of them

new_cols_to_keep = ['age', 'weight', 'num_procedures', 'race', 'gender']

newdata3 = data[new_cols_to_keep]



from sklearn.preprocessing import MinMaxScaler



array = data['num_lab_procedures'].values

array = array.reshape(-1,1)

array

MM = MinMaxScaler() 

MM.fit(array)

newdata3['num_lab_procedures']=MM.transform(array)



array1 = data['time_in_hospital'].values

array1 = array1.reshape(-1,1)

MM1 = MinMaxScaler()

MM1.fit(array1)

newdata3['time_in_hospital']=MM1.transform(array1)



newdata3.head()
newdata3_ohe = pd.get_dummies(newdata3, columns=['age','weight', 'num_procedures'])

newdata3_ohe = newdata3_ohe.drop(['gender','race'],axis=1)



from sklearn.cluster import Birch

brc = Birch(n_clusters=2, threshold=0.65)



pred1 = brc.fit_predict(newdata3_ohe)



np.savetxt("sub2.csv", pred1,header='target') #this is without index column, I appended that from the other files.


