# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")

df_train.head()
#Burada önemli olmayan columnları temizleyeceğiz.

df_train = df_train.drop(columns=["Id"],axis=1)

df_train.head()
print(df_train.info())

print(df_train.describe())
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)



df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max() #just checking that there's no missing data missing...


from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

for column in df_train.columns:

    if df_train[column].dtype == type(object):

        le = LabelEncoder()

        df_train[column] = le.fit_transform(df_train[column].astype(str))

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values=np.nan, strategy='mean')

new_df = imp.fit_transform(df_train)

df_train_new = pd.DataFrame(new_df)

df_train_new.columns = df_train.columns
df_train_new.head()

del df_train

df_train = df_train_new

corr_df = df_train.corr().abs()

corr_df
df_train.head()
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

new_df = ss.fit_transform(df_train)

new_df = pd.DataFrame(new_df)

new_df.columns = df_train.columns
new_df.head()
X = new_df.drop(columns=["SalePrice"])

Y = new_df["SalePrice"]

print("X Shape is : ",X.shape)

Y = Y.values.reshape(-1,1)

print("Y Shape is : ",Y.shape)
mean = Y.mean()

Y = [Y>=mean]

Y = np.array(Y).reshape(-1,1)

print(Y.shape)

Y = np.array([1 if item == True else 0 for item in Y]).reshape(-1,1)

Y.shape
X[:5],Y[:5]
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split



X_train , X_test , Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)

pca = PCA(n_components=10)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)
X_train.shape , X_test.shape , Y_train.shape, Y_test.shape
from sklearn.linear_model import LogisticRegressionCV

Y_train = Y_train.reshape(-1,)



from keras.layers import Input, Dense

from keras.models import Model

from keras import regularizers



# This returns a tensor

inputs = Input(shape=(X_train.shape[1],))



# a layer instance is callable on a tensor, and returns a tensor

x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(inputs)

x = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)

x = Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)

x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)

predictions = Dense(1, activation='sigmoid')(x)



# This creates a model that includes

# the Input layer and three Dense layers

model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.fit(X_train, Y_train,epochs=1000,batch_size=256,shuffle=True)  # starts training
score, acc = model.evaluate(X_test, Y_test,

                            batch_size=256)

print('Test score:', score)

print('Test accuracy:', acc)

predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(predicted.round(),Y_test)
TP , TN , FN,FP = cm[0][0] , cm[0][1] ,cm[1][0] ,cm[1][1] 
print("Class 1 Prediction Accuracy according to CM : {}".format(TP / (TP+TN) * 100))

print("Class 0 Prediction Accuracy According to CM : {}".format(FP / (FP+FN) * 100))