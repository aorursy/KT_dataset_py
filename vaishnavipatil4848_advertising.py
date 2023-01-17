import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf

%matplotlib inline
X=pd.read_csv('../input/advertising/advertising.csv')
X.head()
X.dropna(axis=0,subset=['Clicked on Ad'],inplace=True)

y=X['Clicked on Ad']

X=X.drop('Clicked on Ad',axis=1)
plt.figure(figsize=(10,8))

sns.lmplot(x='Daily Time Spent on Site',y='Age',data=X,hue='Male',palette='plasma')
sns.lineplot(x='Age',y='Area Income',data=X,hue='Male')
plt.figure(figsize=(10,8))

sns.jointplot('Daily Internet Usage','Daily Time Spent on Site',data=X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()



object_cols=[col for col in X_train.columns if

            X_train[col].dtype=="object"]

good_cols=[col for col in object_cols if

          set(X_train[col])==set(X_test[col])]

bad_cols=list(set(object_cols)-set(good_cols))
label_train=X_train.drop(bad_cols,axis=1)

label_test=X_test.drop(bad_cols,axis=1)
label_train.head()
for col in good_cols:

    label_train[col]=pd.DataFrame(encoder.fit_transform(label_train[col]))

    label_test[col]=pd,DataFrame(encoder.fit_transform(label_test[col]))
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaled_features=scaler.fit_transform(label_train)

scaled_features1=scaler.fit_transform(label_test)
X_feat_train=pd.DataFrame(scaled_features,columns=label_train.columns)

X_feat_test=pd.DataFrame(scaled_features1,columns=label_test.columns)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_feat_train,y_train)

preds=knn.predict(X_feat_test)
from sklearn.metrics import classification_report,confusion_matrix
print("Classification Report:",classification_report(y_test,preds))

print("Confusion Matrix:",confusion_matrix(y_test,preds))
error_rate=[]

for i in range(1,40):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_feat_train,y_train)

    preds_i=knn.predict(X_feat_test)

    error_rate.append(np.mean(preds_i != y_test))
plt.figure(figsize=(9,7))

plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize='10')

plt.title('Error_rate vs K')

plt.xlabel('k')

plt.ylabel('error_rate')
knn=KNeighborsClassifier(n_neighbors=13)

knn.fit(X_feat_train,y_train)

preds=knn.predict(X_feat_test)

print(classification_report(y_test,preds))

print(confusion_matrix(y_test,preds))
X_feat_train.shape
n,d=X_feat_train.shape
model=tf.keras.models.Sequential([

    tf.keras.layers.Input(shape=(d,)),

    tf.keras.layers.Dense(1,activation='sigmoid')

])
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
r=model.fit(X_feat_train,y_train,validation_data=(X_feat_test,y_test),epochs=100)
#evaluate data

print("Train error=",model.evaluate(X_feat_train,y_train))

print("Test error=",model.evaluate(X_feat_test,y_test))
plt.plot(r.history['loss'],label='loss')

plt.plot(r.history['val_loss'],label='val_loss')

plt.legend()
plt.plot(r.history['accuracy'],label='acc')

plt.plot(r.history['val_accuracy'],label='val_acc')

plt.legend()