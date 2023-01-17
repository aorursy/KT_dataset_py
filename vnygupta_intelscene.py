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
data=pd.read_csv('../input/scene-classification/train-scene classification/train.csv')
#data['sort'] = data['image_name'].str.extract('(\d+)', expand=False).astype(str)
#data.sort_values('sort',inplace=True, ascending=True)

#data = data.drop('sort', axis=1)
data.label=data.label.astype(str)
from keras.preprocessing import  image
from sklearn.model_selection import train_test_split

X_train,X_test=train_test_split(data,test_size=0.01,random_state=42)
scene=[]

for i in range(len(data)):

    

    scene.append(data.iloc[:,1][i])
pd.Series(scene).value_counts()
dir1='../input/scene-classification/train-scene classification/train/'
datagen=image.ImageDataGenerator(rescale=1./255,shear_range=0.3,zoom_range=0.4,horizontal_flip=True,brightness_range=(0.1,0.6),height_shift_range=0.3,rotation_range=0.4,vertical_flip=0.4,width_shift_range=0.5,zca_whitening=0.3)
train_gen=datagen.flow_from_dataframe(dataframe=X_train,directory=dir1,batch_size=12,class_mode="categorical",x_col="image_name",color_mode="rgb", y_col="label",target_size=(224,224))
train_gen.class_indices
list1=list(X_test['image_name'])

list2=list(X_test['label'])

list3=[list1,list2]
X_test=pd.DataFrame(list3).T
X_test.columns=['image_name','label']
X_test
val_gen=image.ImageDataGenerator(rescale=1./255)
val_gen=val_gen.flow_from_dataframe(dataframe=X_test,directory=dir1,batch_size=12,class_mode="categorical",x_col="image_name",color_mode="rgb", y_col="label",target_size=(224,224))
'''for i in range(len(image_data)):

    image_data[i]=image_data[i]/255'''
from keras import layers

from keras import models

from keras import optimizers
# Final Model Architecture:



kernel=(3,3)

act='relu'

modelN = models.Sequential()

modelN.add(layers.Conv2D(32, kernel, padding='same',input_shape=(224,224,3)))

modelN.add(layers.Activation(act))

#

modelN.add(layers.Conv2D(32, kernel, padding='same' ))

modelN.add(layers.BatchNormalization())

modelN.add(layers.Activation(act))

modelN.add(layers.Dropout(0.1))

#

modelN.add(layers.Conv2D(64, kernel, padding='same'))

modelN.add(layers.BatchNormalization())

modelN.add(layers.Activation(act))

modelN.add(layers.Dropout(0.1))

#

modelN.add(layers.Conv2D(64, kernel, padding='same'))

modelN.add(layers.BatchNormalization())

modelN.add(layers.Activation(act))

modelN.add(layers.MaxPooling2D(pool_size=(2, 2)))

modelN.add(layers.Dropout(0.1))

#

modelN.add(layers.Conv2D(128, kernel, padding='same'))

modelN.add(layers.BatchNormalization())

modelN.add(layers.Activation(act))

modelN.add(layers.MaxPooling2D(pool_size=(2, 2)))

modelN.add(layers.Dropout(0.1))

#

modelN.add(layers.Conv2D(128, kernel, padding='same'))

modelN.add(layers.BatchNormalization())

modelN.add(layers.Activation(act))

modelN.add(layers.MaxPooling2D(pool_size=(2, 2)))

modelN.add(layers.Dropout(0.1))

#

modelN.add(layers.Conv2D(256, kernel, padding='same'))

modelN.add(layers.BatchNormalization())

modelN.add(layers.Activation(act))

modelN.add(layers.MaxPooling2D(pool_size=(2, 2)))

modelN.add(layers.Dropout(0.1))

#

modelN.add(layers.Conv2D(256, kernel, padding='same'))

modelN.add(layers.BatchNormalization())

modelN.add(layers.Activation(act))

modelN.add(layers.MaxPooling2D(pool_size=(2, 2)))

#

modelN.add(layers.Conv2D(512, kernel, padding='same'))

modelN.add(layers.BatchNormalization())

modelN.add(layers.Activation(act))

modelN.add(layers.MaxPooling2D(pool_size=(2, 2)))

#

modelN.add(layers.Conv2D(512, kernel, padding='same'))

modelN.add(layers.BatchNormalization())

modelN.add(layers.Activation(act))

modelN.add(layers.MaxPooling2D(pool_size=(2, 2)))

modelN.add(layers.Flatten())

## this converts our 3D feature maps to 1D feature vectors

modelN.add(layers.Dense(1096))

modelN.add(layers.BatchNormalization())

modelN.add(layers.Activation(act))

modelN.add(layers.Dropout(0.2))

modelN.add(layers.Dense(1096))

modelN.add(layers.BatchNormalization())

modelN.add(layers.Activation(act))

modelN.add(layers.Dropout(0.2))

modelN.add(layers.Dense(6, activation='softmax'))

#sgd =optimizers.Adam(lr=1.0, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# optimizer:

modelN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

print ('Training....')





#fit

nb_epoch =32

batch_size =512



STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size

STEP_SIZE_VALID=val_gen.n//val_gen.batch_size

modelN.fit_generator(generator=train_gen,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    #validation_split=0.2,

                    validation_data=val_gen,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=64)
modelN.save('intel0318.h5')
from matplotlib import pyplot as plt
plt.plot(modelN.history.history['val_categorical_accuracy']),plt.plot(modelN.history.history['categorical_accuracy'])
data_test=pd.read_csv('../input/scene-classification/test_WyRytb0.csv')
image_data=[]

#image_label=[]

for i in range(len(data_test.iloc[:,0])):

    try:

        image_data.append(image.img_to_array(image.load_img(dir1+data_test.iloc[:,0][i],color_mode="grayscale",target_size=(224,224))))

    #image_label.append(ohe.transform(encoder.transform([data.iloc[:,1][i]]).reshape((1,1))))

    #ohe.transform(encoder.transform([data.iloc[:,1][i]]).reshape((1,1)))

    except:

        print(i)
results=[]

for i in range(len(image_data)):

    image_data[i]=image_data[i]/255

    image_data[i]=np.expand_dims(image_data[i],axis=0)

    result=modelN.predict(image_data[i])

    results.append(result)
best_preds = np.asarray([np.argmax(line) for line in results])
pd.Series(best_preds).value_counts()
data_test['label']=best_preds
data_test.to_csv('submissionv4.csv',index=False)
train_gen.class_indices