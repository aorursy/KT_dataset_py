import pandas as pd

import numpy as np

from matplotlib import pyplot as plt



from keras.models import Sequential, load_model



from keras.layers import Dense, Dropout, Flatten



from keras.layers.convolutional import Conv2D, MaxPooling2D



from keras.utils.np_utils import to_categorical



import seaborn as sns



from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score

data_train = pd.read_csv('../input/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashion-mnist_test.csv')
rows,cols=28,28

labels=10
data_train.label.value_counts()
X_train=data_train.iloc[:,:-1].values

y_train=data_train.loc[:,'label'].values

X_test=data_test.iloc[:,:-1].values

y_test=data_test.loc[:,'label'].values



X_train=X_train.reshape(X_train.shape[0],28,28,1)

X_test=X_test.reshape(X_test.shape[0],28,28,1)



y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(labels, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(x=X_train,y=y_train,batch_size=2000, epochs=10,validation_data=(X_test,y_test))



model.summary()
from sklearn.metrics import accuracy_score as acc

y_pred=model.predict(X_test).argmax(axis=-1)

y_test=y_test.argmax(axis=-1)

acc(y_test,y_pred)
cm = confusion_matrix(y_pred,y_test)

print(cm)

plt.figure(figsize = (12,10))

sns.heatmap(cm, annot=True, cmap="coolwarm")
import random

labels=['top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

plt.figure(figsize=(15,25))

n_test = X_test.shape[0]

for i in range(1,50) :

    ir = random.randint(0,n_test)

    plt.subplot(10,5,i)

    plt.axis('off')

    plt.imshow(X_test[ir].reshape(28,28), cmap="gray_r")

    pred_classe = y_pred[ir]

    plt.title(labels[pred_classe])