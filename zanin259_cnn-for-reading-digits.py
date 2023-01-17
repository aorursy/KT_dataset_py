import warnings

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

from keras.preprocessing.image import ImageDataGenerator
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')



test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



submission=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
train.head()
test.head()
# We check if there are any missing values in the dataset



train.isnull().any().describe()
test.isnull().any().describe()
# We create the X_train dataset after removing the label columns and y_train with the label column.

X_train=train.drop(columns='label')



y_train=train['label']
# We plot the count plot for the label column.

plt.figure(figsize=(8,6))

sns.countplot(y_train)
# We normalize the train and test dataset by 255 



X_train=X_train/255



test=test/255
y_train=to_categorical(y_train,num_classes=10)
# We reshape the dataset in 3 dimesnions



X_train=X_train.values.reshape(-1,28,28,1)



test=test.values.reshape(-1,28,28,1)
# We Split the dataset to train and test set

X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.1,random_state=0)
plt.imshow(X_train[0][:,:,0])
# We Start building our CNN Model by creating an convoltion layer and max pooling layer



classifier = Sequential()



classifier.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))



classifier.add(MaxPooling2D(pool_size=(2,2)))



# We add a second convolution and Max Pooling layer

classifier.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))



classifier.add(MaxPooling2D(pool_size=(2,2)))



classifier.add(Flatten())



classifier.add(Dense(units=128,activation='relu'))



classifier.add(Dense(units=10,activation='softmax'))



classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# We create the image genarator



datagen=ImageDataGenerator(rotation_range=10,zoom_range = 0.1,width_shift_range=0.1,height_shift_range=0.1)



datagen.fit(X_train)
hist= classifier.fit_generator(datagen.flow(X_train,y_train,batch_size=32),epochs=2,validation_data=(X_val,y_val), steps_per_epoch=len(X_train)/32)
hist.history['val_accuracy']
# We plot the acuracy plot 

plt.plot(range(2), hist.history['accuracy'],label='Train_Accuracy')

plt.plot(range(2), hist.history['val_accuracy'],label='Val_Accuracy')

plt.legend()
plt.plot(range(2), hist.history['loss'],label='Train_Loss')

plt.plot(range(2), hist.history['val_loss'],label='Val_Loss')

plt.legend()
# We predict for the test dataset

y_pred=classifier.predict(test)



y_pred=np.argmax(y_pred,axis=1)
# We convert it to dataframe

y_pred=pd.DataFrame(y_pred,columns=['Label'])
submission['Label']=y_pred
submission.head()
submission.to_csv('Submission.csv',index=False)