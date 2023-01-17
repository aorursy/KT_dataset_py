# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

base="../input"

train_in="../input/train/train"

file_names=(os.listdir("../input/train/train"))

# Any results you write to the current directory are saved as output.
#print(file_names)

targets=list()

full_paths=list()

for x in file_names:

    temp=os.path.join(train_in,x)

    full_paths.append(temp)

    #temp2=file_names.split(".")[0] #NOTERROR: TAKE A SINGLE ITEM INSTEAD OF ENTIRE LIST

    temp2=x.split(".")[0]

    targets.append(temp2)

print(targets[:10])
#targets2=list(len(targets))

#for y in range(len(targets)):

#    if targets[y] == 'dog':

#        targets[y]=1

##    else:

 #       targets[y]=0
print(full_paths[:10])

print(targets[:10])
dataset=pd.DataFrame()

dataset['Paths']=full_paths

dataset['Name']=targets
dataset.head()
#import matplotlib.pyplot as plt

#plt.imshow(dataset.iloc(0))
#from sklearn import train_test_split

import sklearn

from sklearn.model_selection import train_test_split

train_1230,test_123=sklearn.model_selection.train_test_split(dataset,test_size=0.2,random_state=42)
import tensorflow as tf
model=tf.keras.Sequential([tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)),

                          tf.keras.layers.MaxPooling2D(2,2),

                          tf.keras.layers.Conv2D(128,(3,3),activation='relu'),

                          tf.keras.layers.MaxPooling2D(2,2),

                          tf.keras.layers.Conv2D(128,(3,3),activation='relu'),

                          tf.keras.layers.MaxPooling2D(2,2),

                          tf.keras.layers.Conv2D(64,(3,3), activation='relu'),

                          tf.keras.layers.MaxPooling2D(2,2),

                          tf.keras.layers.Flatten(),

                          tf.keras.layers.Dense(512,activation='relu'),

                          tf.keras.layers.Dense(1,activation='sigmoid')])
model.summary()
from tensorflow.keras.optimizers import RMSprop



model.compile(optimizer=RMSprop(lr=0.001),

              loss="binary_crossentropy",metrics=['acc'])

print("[INFO]: model compiled...")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255)

test_datagen=ImageDataGenerator(rescale=1./255)
train_data_generator=train_datagen.flow_from_dataframe(dataframe=train_1230,

                                                      x_col='Paths',

                                                      y_col='Name',target_size=(150,150),

                                                      class_mode="binary",

                                                      batch_size=150)
print(train_data_generator)
test_data_generator=test_datagen.flow_from_dataframe(dataframe=test_123,

                                                      x_col='Paths',

                                                      y_col='Name',

                                                      target_size=(150,150),

                                                      class_mode="binary",

                                                      batch_size=150)
modelHistory=model.fit_generator(train_data_generator,

                                epochs=10,

                                validation_data=test_data_generator,

                                validation_steps=test_123.shape[0]//150,

                                steps_per_epoch=train_1230.shape[0]//150)
#history=model.fit_generator(train_data_generator,validation_data=test_data_generator,

#                           epochs=1)#,

#                            #validation_steps=test.shape[0]//150,

#                            #steps_per_epoch=train.shape[0]//150)

#                                #steps_per_epoch=200)
model.save_weights("models.h5")
base2="../input/test1/test1"

test_files_name=os.listdir("../input/test1/test1")

#print(test_files_name)

full_paths2=list()

for x in test_files_name:

    temp=os.path.join(base2,x)

    full_paths2.append(temp)

    #temp2=file_names.split(".")[0] #NOTERROR: TAKE A SINGLE ITEM INSTEAD OF ENTIRE LIST

    #temp2=x.split(".")[0]

    #targets.append(temp2)

full_paths2[:10]
test_df=pd.DataFrame({'filename':full_paths2})

no_sample=(test_df.shape[0])

print(no_sample)

test_df.head()
test1gen=ImageDataGenerator(rescale=1./255)

test1_image_generator=test1gen.flow_from_dataframe(dataframe=test_df,x_col='filename',y_col=None, class_mode =None, target_size=(150,150),batch_size=15)
predict = model.predict_generator(test1_image_generator, steps=np.ceil(no_sample/15))
print(predict)
print(len(predict))

print(len(test_files_name))
submission_name_list=list()

for x in test_files_name:

    t=x.split(".")[0]

    submission_name_list.append(t)

print(submission_name_list[:10])
print(type(predict))

list_temp=list()

list_temp=predict
threshold=0.5

class_np=np.where(predict > threshold, 1,0)
class_np[:10]
#threshold = 0.5

#test_cat=pd.DataFrame()

#test_cat['id'] = list_temp

#test_cat['category'] = np.where(test_df['probability'] > threshold, 1,0)

#threshold = 0.5

test_cat=pd.DataFrame()

test_cat['id'] = submission_name_list

test_cat['label'] = class_np
test_cat.head()
test_cat['label'].value_counts().plot.bar()
test_cat.to_csv('submission.csv', index=False)