# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
         os.path.join(dirname, filename)

# Any results you write to the current directory are saved as output.
import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import keras
from keras import regularizers



from keras import layers

from matplotlib.image import imread
import matplotlib.pyplot as plt
%matplotlib inline
SEED = 257

TRAIN_DIR = '/kaggle/input/hotdogs-spbu/train/train' 
TEST_DIR = '/kaggle/input/hotdogs-spbu/test/test'  

categories = ['hot dog', 'not hot dog']
X, y = [], []

for category in categories:
    category_dir = os.path.join(TRAIN_DIR, category)
    for image_path in os.listdir(category_dir):
        X.append(imread(os.path.join(category_dir, image_path)))  
        y.append(category)  
y = [1 if x == 'hot dog' else 0 for x in y]  
plt.axis("off");
plt.imshow(X[9])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255,
                                  rotation_range=40,
                                  width_shift_range=0.2,      
                                  height_shift_range=0.2,
                                  brightness_range=(0.6, 1),
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  vertical_flip=True)



train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=(100, 100),
                                                    batch_size=200,
                                                    class_mode='binary'
                                                   )
y_create=[]
X_create=[]

for i in range(51):
    
    for outcome in train_generator[i][1]:
        y_create.append(outcome)
    for image in train_generator[i][0]:
        X_create.append(image)
plt.axis("off");
plt.imshow(X_create[59])
#append new images

X_mix = X[:5026] + X_create[5026:]
y_mix = y[:5026] + y_create[5026:]

X_Mix = X+X_create
y_Mix = y+y_create

X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.25, random_state=SEED)

X_mix_train, X_mix_test, y_mix_train, y_mix_test = train_test_split(np.array(X_mix), np.array(y_mix), test_size=0.25, random_state=SEED)

X_Mix_train, X_Mix_test, y_Mix_train, y_Mix_test = train_test_split(np.array(X_Mix), np.array(y_Mix), test_size=0.25, random_state=SEED)
covn_base_VGG=keras.applications.VGG16(weights='imagenet',include_top=False,input_shape=(100,100,3) )


covn_base_VGG.trainable=False
model_VGG=keras.Sequential()

model_VGG.add( covn_base_VGG )
model_VGG.add( layers.GlobalAveragePooling2D() )
model_VGG.add( layers.Dense(522,activation='relu') )
model_VGG.add( layers.Dropout(0.5) )

model_VGG.add( layers.Dense(1,activation='sigmoid') )

model_VGG.compile(optimizer=keras.optimizers.Adam(lr=0.0005),loss='binary_crossentropy',metrics=['acc'])
history_VGG=model_VGG.fit(X_train, y_train, batch_size=200, epochs=20, validation_data=(X_test, y_test))
predict=model_VGG.predict_proba(np.array(X))


r_VGG=roc_auc_score(np.array(y), predict)
r_VGG


#buid Xception model

co_Xception=keras.applications.xception.Xception(weights='imagenet'
                                                        ,include_top=False
                                                        ,input_shape=(100,100,3)
                                                        ,pooling='avg' )

co_Xception.trainable=True


model_Xception0=keras.Sequential()

model_Xception0.add( co_Xception )


model_Xception0.add( layers.Dense(522,activation='relu') )


model_Xception0.add( layers.Dropout(0.5) )



model_Xception0.add( layers.Dense(1,activation='sigmoid') )




model_Xception0.compile(optimizer=keras.optimizers.Adam(lr=0.0005),loss='binary_crossentropy',metrics=['acc'])
history_Xception0=model_Xception0.fit(X_train, y_train, batch_size=200, epochs=20, validation_data=(X_test, y_test))
predict=model_Xception0.predict_proba(np.array(X))


r0=roc_auc_score(np.array(y), predict)
r0


covn_base_Xception=keras.applications.xception.Xception(weights='imagenet'
                                                        ,include_top=False
                                                        ,input_shape=(100,100,3)
                                                        ,pooling='avg' )

#Train model in X_mix

covn_base_Xception.trainable=True

model_Xception=keras.Sequential()

model_Xception.add( covn_base_Xception )


model_Xception.add( layers.Dense(522,activation='relu') )


model_Xception.add( layers.Dropout(0.5) )



model_Xception.add( layers.Dense(1,activation='sigmoid') )




model_Xception.compile(optimizer=keras.optimizers.Adam(lr=0.0005),loss='binary_crossentropy',metrics=['acc'])
history_Xception=model_Xception.fit(X_mix_train, y_mix_train, batch_size=200, epochs=20, validation_data=(X_mix_test, y_mix_test))
predict=model_Xception.predict_proba(np.array(X))


r=roc_auc_score(np.array(y), predict)
r

#train model in X_Mix
covn_base_Xception1=keras.applications.xception.Xception(weights='imagenet'
                                                        ,include_top=False
                                                        ,input_shape=(100,100,3)
                                                        ,pooling='avg' )

covn_base_Xception1.trainable=True

model_Xception1=keras.Sequential()

model_Xception1.add( covn_base_Xception1 )


model_Xception1.add( layers.Dense(522,activation='relu') )


model_Xception1.add( layers.Dropout(0.5) )



model_Xception1.add( layers.Dense(1,activation='sigmoid') )




model_Xception1.compile(optimizer=keras.optimizers.Adam(lr=0.0005),loss='binary_crossentropy',metrics=['acc'])
history_Xception1=model_Xception1.fit(X_Mix_train, y_Mix_train, batch_size=200, epochs=20, validation_data=(X_Mix_test, y_Mix_test))
predict=model_Xception1.predict_proba(np.array(X))


r1=roc_auc_score(np.array(y), predict)
r1
loss = history_Xception0.history['loss']
val_loss = history_Xception0.history['val_loss']
acc=history_Xception0.history['acc']
val_acc = history_Xception0.history['val_acc']


plt.plot(loss, label='loss',c='blue',linestyle="--")
plt.plot(val_loss, label='val_loss',c='blue' )
plt.plot(acc, label='acc',c='r',linestyle="--")
plt.plot(val_acc, label='val_acc',c='r')

plt.title('model_Xception0')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend( )
plt.savefig('./loss.png')

loss = history_Xception.history['loss']
val_loss = history_Xception.history['val_loss']
acc=history_Xception.history['acc']
val_acc = history_Xception.history['val_acc']


plt.plot(loss, label='loss',c='blue',linestyle="--")
plt.plot(val_loss, label='val_loss',c='blue' )
plt.plot(acc, label='acc',c='r',linestyle="--")
plt.plot(val_acc, label='val_acc',c='r')

plt.title('model_Xception')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend( )
plt.savefig('./loss.png')

loss = history_Xception1.history['loss']
val_loss = history_Xception1.history['val_loss']
acc=history_Xception1.history['acc']
val_acc = history_Xception1.history['val_acc']


plt.plot(loss, label='loss',c='blue',linestyle="--")
plt.plot(val_loss, label='val_loss',c='blue' )
plt.plot(acc, label='acc',c='r',linestyle="--")
plt.plot(val_acc, label='val_acc',c='r')

plt.title('model_Xception1')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('./loss.png')




covn_base_Xception2=covn_base_Xception1


covn_base_Xception2.trainable=False
model_Xception2=keras.Sequential()
model_Xception2.add( covn_base_Xception2_ )


model_Xception2.add( layers.Dense(1024,kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),activation='relu') )
model_Xception2.add( layers.Dense(522,kernel_regularizer=regularizers.l2(0.1),activation='relu') )

model_Xception2.add( layers.Dropout(0.5) )

model_Xception2.add( layers.Dense(1,activation='sigmoid') )

model_Xception2.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss='binary_crossentropy',metrics=['acc'])
history_Xception2=model_Xception2.fit(X_Mix_train, y_Mix_train, batch_size=200, epochs=12, validation_data=(X_Mix_test, y_Mix_test))

for i in covn_base_Xception2.layers[: -30]:
    
    layers.trainable=False
    
model_Xception2.compile(optimizer=keras.optimizers.Adam(lr=0.0001/10),loss='binary_crossentropy',metrics=['acc'])
history_Xception2=model_Xception2.fit(X_Mix_train, y_Mix_train, batch_size=200, epochs=12, validation_data=(X_Mix_test, y_Mix_test))

model_Xception2.compile(optimizer=keras.optimizers.Adam(lr=0.0001/20),loss='binary_crossentropy',metrics=['acc'])

history_Xception2=model_Xception2.fit(X_Mix_train, y_Mix_train, batch_size=200, epochs=12, validation_data=(X_Mix_test, y_Mix_test))


predict=model_Xception2.predict_proba(np.array(X))
r2=roc_auc_score(np.array(y), predict)
r2
loss = history_Xception1.history['loss']
val_loss = history_Xception1.history['val_loss']
acc=history_Xception1.history['acc']
val_acc = history_Xception1.history['val_acc']


plt.plot(loss, label='loss',c='blue',linestyle="--")
plt.plot(val_loss, label='val_loss',c='blue' )
plt.plot(acc, label='acc',c='r',linestyle="--")
plt.plot(val_acc, label='val_acc',c='r')

plt.title('model_Xception1')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('./loss.png')

leaderboard_X = []
leaderboard_filenames = []

for image_path in os.listdir(TEST_DIR):
    leaderboard_X.append(imread(os.path.join(TEST_DIR, image_path))) #save the information of images(test)
    leaderboard_filenames.append(image_path) #save the documents name of images(test)

    
plt.axis("off");
plt.imshow(leaderboard_X[0])
leadeboard_predictions = []

for x in leaderboard_X:
    leadeboard_predictions.append(model_Xception2.predict_proba (np.expand_dims(x,axis=0)))
    
    
    

Leadeboard_predictions = []


for x in leadeboard_predictions:
    Leadeboard_predictions.append( 
                                 round(x[0][0],2)
                                 )
font = {
    'family': 'serif',
    'color':  'darkred',
    'weight': 'bold',
    'size': 22,
}


idx = 378

plt.axis("off");



if Leadeboard_predictions[idx] > 0.5:
    plt.text(20, -5, 'HOT DOG!!!', fontdict=font)
else:
    plt.text(15, -5,'not hot dog...', fontdict=font)
plt.imshow(leaderboard_X[idx])
submission = pd.DataFrame(
    {
        'image_id':leaderboard_filenames, 
        'image_hot_dog_probability': Leadeboard_predictions
    }
)



submission.head()




submission.to_csv('submit.csv', index=False) # save the predict to csv































































