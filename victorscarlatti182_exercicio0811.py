

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nomes = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"]

df = pd.read_csv("/kaggle/input/creditscreening/credit.data", sep=",", names=nomes, na_values='?')

df.head()
df.describe()
df.isnull().sum()


values = {'A1': df['A1'].mode(),'A4': df['A4'].mode(),

         'A5': df['A5'].mode(), 'A6': df['A6'].mode(),

         'A7': df['A7'].mode() }

df.fillna(value=values,inplace=True)
cols = ['A1', 'A4','A5','A6','A7']

for i in cols:

    df[i].fillna(value=df[i].mode, inplace=True)
df.isnull().sum()
values2 = {'A2': df['A2'].mean() , 'A14': df['A14'].mean()}



df.fillna(value=values2)

cols = ['A2', 'A14']

for i in cols:

    df[i].fillna(value=df[i].mode, inplace=True)
df.isnull().sum()
df['A1']=df['A1'].astype(str)

df['A1']=df['A1'].astype('category')

df['A4']=df['A4'].astype(str)

df['A4']=df['A4'].astype('category')

df['A5']=df['A5'].astype(str)

df['A5']=df['A5'].astype('category')

df['A6']=df['A6'].astype(str)

df['A6']=df['A6'].astype('category')

df['A7']=df['A7'].astype(str)

df['A7']=df['A7'].astype('category')

df['A9']=df['A9'].astype(str)

df['A9']=df['A9'].astype('category')

df['A10']=df['A10'].astype(str)

df['A10']=df['A10'].astype('category')

df['A12']=df['A12'].astype(str)

df['A12']=df['A12'].astype('category')

df['A13']=df['A13'].astype(str)

df['A1']=df['A13'].astype('category')





df['A16']=df['A16'].astype('category')

df.info()



df["A16"].cat.codes
x = df.loc[:,'A1':'A15']

y = df.loc[:,'A16':'A16']
x = pd.get_dummies( x, columns=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])
x.head()


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

Xss = ss.fit_transform(x)

Xss


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Xss,y, test_size=0.2)
y_train.shape


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

modelo = knn.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

y_score = modelo.score(X_test, y_test)

y_score