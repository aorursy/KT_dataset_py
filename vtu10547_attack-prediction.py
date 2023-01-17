#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sns
df=pd.read_csv("../input/attacked/attack.csv")

df.head()
sns.countplot(df["attack"])
df.dtypes
df.dropna(axis=0,inplace=True)
df.astype('int64')
df[" FlowPackets"]=pd.to_numeric(df[" FlowPackets"],errors="coerce")
#Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df[['FlowDuration']]=sc.fit_transform(df[['FlowDuration']])
df[[' FlowPackets']]=sc.fit_transform(df[[' FlowPackets']])
df.head()
X=df[["Port","FlowDuration"," FlowPackets"]]
y=df["attack"]
df.isnull().sum()
df.dropna(axis=0,inplace=True)
df.shape
df.dtypes
df.isnull().sum()
y.shape
X.head()
y.head()
#splitting the dataset in trainset and testset 
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=10)
x_train.shape
x_train
x_train

from sklearn.preprocessing import MinMaxScaler

sc=MinMaxScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
x_train


#Importing libraries for CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,Dropout,GlobalAveragePooling1D,MaxPooling1D,Dense

#Initializing the model 
model_m = Sequential()
model_m.add(Dense(10,kernel_initializer= 'uniform',activation= 'relu',input_dim=3)) 
model_m.add(Dense(10,kernel_initializer= 'uniform',activation= 'relu')) 
#model_m.add(MaxPooling1D())
#model_m.add(Dropout(0.5))
model_m.add(Dense(1, activation='sigmoid'))
print(model_m.summary())
model_m.compile(optimizer="adam",loss="binary_crossentropy",metrics=['Accuracy'])

history=model_m.fit(x_train,y_train,batch_size=2,epochs=20,validation_data=(x_test, y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
model=MLPClassifier()
log=LogisticRegression()
model.fit(x_train,y_train)

log.fit(x_train,y_train)
y_hat=model.predict(x_test)
y_hat1=log.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,y_hat)
print(accuracy_score(y_test,y_hat1))
confusion_matrix(y_test,y_hat)
dd=model_m.predict(x_test)
dd
dd=[dd>0.5]
dd

len(dd[0])
y_test=np.asarray(y_test)

y_test.shape
accuracy_score(y_test,dd[0])*100

