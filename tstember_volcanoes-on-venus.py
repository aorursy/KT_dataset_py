# Venus volcanoes recognizer inspired by LeNet5 and using Keras
import numpy as np
import pandas as pd
import tensorflow as tf 


from sklearn.linear_model import LogisticRegression


import keras 
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation, Dropout
from keras.optimizers import Adam

from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import mean_squared_error


import seaborn as sns
X_train = pd.read_csv('../input/volcanoes_train/train_images.csv', header=None, index_col=None)
X_test = pd.read_csv('../input/volcanoes_test/test_images.csv', header=None, index_col=None)

Y_train = pd.read_csv('../input/volcanoes_train/train_labels.csv')
Y_test = pd.read_csv('../input/volcanoes_test/test_labels.csv')

X_train.head()
Y_train.head()
print (X_train.shape)
print (X_test.shape)
print (Y_train.shape)
print (Y_test.shape)
#a lot of nan values for type, radius and number volcanoes for images with no volcano
Y_train.fillna(value=0,inplace=True)
Y_test.fillna(value=0,inplace=True)
print(sum(X_train.isna().sum()))
print(sum(X_test.isna().sum()))

print(Y_train.isna().sum())
print(Y_test.isna().sum())
plt.hist(Y_train["Volcano?"])
plt.show()
plt.hist(Y_train["Type"])
plt.show()
plt.hist(Y_train["Radius"])
plt.show()
plt.hist(Y_train["Number Volcanoes"])
plt.show()
X_train=np.array(X_train)
X_test=np.array(X_test)
#Rescale data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255.
X_test = X_test/255.
X_train[0].shape
#Reshape datasets for CNN
X_train_CNN = X_train.reshape(X_train.shape[0], 110, 110, 1)
X_test_CNN = X_test.reshape(X_test.shape[0], 110, 110, 1)
Y_train_volcano = Y_train['Volcano?']
Y_test_volcano = Y_test['Volcano?']
print (Y_train_volcano.shape)
print (Y_test_volcano.shape)
model = Sequential()

#Conv layer 1 
#input 32x32x1, output 28x28x6
model.add(Conv2D(32,(5,5), padding = 'Same', activation = 'relu', input_shape = (110,110,1)))
model.add(BatchNormalization())

#Pooling layer 1
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

#Conv layer 2
model.add(Conv2D(32, (3,3), padding = 'Same', activation = 'relu'))
model.add(BatchNormalization())

#Pooling layer 2
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

#Flatten
model.add(Flatten())

#Fully connected layer 1
model.add(Dense(128, activation = 'relu'))

#Output Layer
model.add(Dense(units = 1,kernel_initializer="uniform", activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Split the train dataset to training and  validation 

X_train_CNN, X_val_CNN, Y_train_volcano, Y_val_volcano = train_test_split(X_train_CNN, Y_train_volcano, test_size = 0.1, random_state=2)
#run model
model.fit(X_train_CNN, Y_train_volcano, batch_size=100, epochs = 15, validation_data=(X_val_CNN, Y_val_volcano))

Y_pred_volcano = model.predict_classes(X_test_CNN)
Y_pred_volcano = pd.DataFrame({'Volcano?': Y_pred_volcano.flatten()})
Y_test_volcano = pd.DataFrame({'Volcano?': Y_test_volcano})
print ("test accuracy: %s " %accuracy_score(Y_test_volcano, Y_pred_volcano))
print ("precision: %s " %precision_score(Y_test_volcano, Y_pred_volcano))
print ("recall: %s " %recall_score(Y_test_volcano, Y_pred_volcano))
print ("f1 score: %s " %f1_score(Y_test_volcano, Y_pred_volcano))


cm = confusion_matrix(Y_test_volcano, Y_pred_volcano)
sns.heatmap(cm,annot=True)
#classification according to type 
#Take the old data before processing X_train for  CNN



X_train_new = X_train[Y_train['Volcano?']==1]
Y_train_clf_type = Y_train[Y_train['Volcano?']==1].drop(['Volcano?','Radius', 'Number Volcanoes'], axis=1)
Y_train_clf_type=np.array(Y_train_clf_type)

X_test_new = X_test[Y_pred_volcano['Volcano?']==1]
Y_test_clf_type = Y_test[Y_pred_volcano['Volcano?']==1].drop(['Volcano?','Radius','Number Volcanoes'], axis=1)
Y_test_clf_type=np.array(Y_test_clf_type)

# LogisticRegression 
logreg = LogisticRegression(C=5000, solver='newton-cg', multi_class='multinomial',max_iter=1000)
logreg.fit(X_train_new, Y_train_clf_type.ravel())
Y_pred_clf_type = logreg.predict(X_test_new)


print ("score: %s " %logreg.score(X_train_new, Y_train_clf_type))
print ("accuracy score: %s " %accuracy_score(Y_test_clf_type, Y_pred_clf_type))
Y_pred_clf_type = pd.DataFrame({'Type': Y_pred_clf_type})
Y_pred_type=[]
j=0
for i in range(Y_pred_volcano.shape[0]):
    if Y_pred_volcano['Volcano?'].iloc[i]==1.0:
        Y_pred_type.append(Y_pred_clf_type['Type'].iloc[j])
        j=j+1
    else:
        Y_pred_type.append(0.0)



Y_pred_type = pd.DataFrame({'Type': Y_pred_type})
Y_pred_type.head()
accuracy_score(Y_test['Type'], Y_pred_type)
#classification according to number of volcanoes
#Take the old data before processing X_train for  CNN


Y_train_clf_num = Y_train[Y_train['Volcano?']==1].drop(['Volcano?','Radius','Type'], axis=1)
Y_train_clf_num=np.array(Y_train_clf_num)


Y_test_clf_num = Y_test[Y_pred_volcano['Volcano?']==1].drop(['Volcano?','Radius','Type'], axis=1)
Y_test_clf_num=np.array(Y_test_clf_num)

logreg = LogisticRegression(C=5000, solver='newton-cg', multi_class='multinomial',max_iter=10000)
logreg.fit(X_train_new, Y_train_clf_num.ravel())
Y_pred_clf_num = logreg.predict(X_test_new)



print ("score: %s " %logreg.score(X_train_new, Y_train_clf_num))
print ("accuracy score: %s " %accuracy_score(Y_test_clf_num, Y_pred_clf_num))
Y_pred_clf_num = pd.DataFrame({'Number Volcanoes': Y_pred_clf_num})
Y_pred_num=[]
j=0
for i in range(Y_pred_volcano.shape[0]):
    if Y_pred_volcano['Volcano?'].iloc[i]==1.0:
        Y_pred_num.append(Y_pred_clf_num['Number Volcanoes'].iloc[j])
        j=j+1
    else:
        Y_pred_num.append(0.0)



Y_pred_num = pd.DataFrame({'Number Volcanoes': Y_pred_num})
Y_pred_num.head()


accuracy_score(Y_test['Number Volcanoes'], Y_pred_num)
#regression for radius 

Y_train_reg = Y_train[Y_train['Volcano?']==1].drop(['Volcano?','Type','Number Volcanoes'], axis=1)
Y_test_reg = Y_test[Y_pred_volcano['Volcano?']==1].drop(['Volcano?','Type','Number Volcanoes'], axis=1)
Y_train_reg=np.array(Y_train_reg)
Y_test_reg=np.array(Y_test_reg)

reg = GradientBoostingRegressor(random_state=20, n_estimators=500)
reg = reg.fit(X_train_new, Y_train_reg.ravel())


print (reg.score(X_train_new, Y_train_reg))

Y_pred_reg = reg.predict(X_test_new)
print (mean_squared_error(Y_test_reg, Y_pred_reg))

Y_pred_reg = pd.DataFrame({'Radius': Y_pred_reg})

Y_pred_rad=[]
j=0
for i in range(Y_pred_volcano.shape[0]):
    if Y_pred_volcano['Volcano?'].iloc[i]==1.0:
        Y_pred_rad.append(Y_pred_reg['Radius'].iloc[j])
        j=j+1
    else:
        Y_pred_rad.append(0.0)

Y_pred_rad = pd.DataFrame({'Radius': Y_pred_rad})
Y_pred_rad.head()
Y_pred_volcano=Y_pred_volcano.join(Y_pred_type)
Y_pred_volcano=Y_pred_volcano.join(Y_pred_rad)
Y_pred_volcano=Y_pred_volcano.join(Y_pred_num)
Y_pred_volcano.head(20)
Y_test.head(20)
