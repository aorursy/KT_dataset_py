# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/data/"))



# Any results you write to the current directory are saved as output.
path='../input/data/Data/'

Data1_path=path+'TrainingData/'

Data2_path=path+'TrainingData2/'

def read_data():

    Data1=pd.read_csv(path+'Train_label.csv')

    Data2=pd.read_csv(path+'DataInfo.csv')

    

    Data1['ImageName']=Data1['ImageName'].apply(lambda x:eval(x))

    Data1['Grading']=Data1['Grading'].apply(lambda x:eval(x))

    Data1['Staging']=Data1['Staging'].apply(lambda x:eval(x))



    Data2['ImageName']=Data2['ImageName'].apply(lambda x:eval(x)+'.jpg')

    Data2['Grading']=Data2['Grading'].apply(lambda x:eval(x))

    Data2['Staging']=Data2['Staging'].apply(lambda x:eval(x))

    

    return pd.concat([Data1,Data2],axis=0)
row_df=read_data()
#print(row_df)
from keras.preprocessing.image import ImageDataGenerator

from keras import models

from keras.layers import Input,Dense,BatchNormalization,GlobalAveragePooling2D,Dropout,Activation,GlobalMaxPooling2D,Concatenate

from keras import optimizers

from sklearn.model_selection import KFold,StratifiedKFold,train_test_split

from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
X=row_df['ImageName'].values

y1=row_df['Grading'].values

y2=row_df['Staging'].values



data_all=row_df.values

y_indice=np.array(range(0,1320))



y1_label=np.zeros(shape=(1320,))

y2_label=np.zeros(shape=(1320,))



i=0

for item in y1:

    if item=='High':

        y1_label[i]=1

    i=i+1



i=0

for item in y2:

    if item=='MIBC':

        y2_label[i]=1

    i=i+1
print(y1_label,y1)

print(y2_label,y2)

print(y1.shape)

print(y2.shape)
X_train,X_test,y_train_indice,y_test_indice=train_test_split(X,y_indice,test_size=0.2,random_state=0)



train_df=pd.DataFrame(data_all[y_train_indice],columns=['ImageName','Grading','Staging'])

train_df['indice']=y_train_indice



test_df=pd.DataFrame(data_all[y_test_indice],columns=['ImageName','Grading','Staging'])

test_df['indice']=y_test_indice
print(train_df,test_df)
train_dataGen=ImageDataGenerator(rescale=1./255,

                                     #rotation_range=90,

                                     #width_shift_range=0.2,

                                     #height_shift_range=0.2,

                                     #shear_range=0.2,

                                     #zoom_range=0.2,

                                     #horizontal_flip=True,

                                     #vertical_flip=True

                                )



valid_dataGen=ImageDataGenerator(rescale=1./255)



train_indice_generator=train_dataGen.flow_from_dataframe(dataframe=train_df,

                                                      directory=Data1_path,

                                                      x_col='ImageName',

                                                      y_col='indice',

                                                      target_size=(299,299),

                                                      class_mode='other',

                                                      color_mode='grayscale',

                                                      batch_size=32)

    

valid_indice_generator=valid_dataGen.flow_from_dataframe(dataframe=test_df,

                                                      directory=Data1_path,

                                                      x_col='ImageName',

                                                      y_col='indice',

                                                      target_size=(299,299),

                                                      class_mode='other',

                                                      color_mode='grayscale',

                                                      batch_size=32)
def trainGen():

    while True:

        X,y_indice=next(train_indice_generator)

        Y=[y1_label[y_indice],y2_label[y_indice]]

        yield X,Y

def validGen():

    while True:

        X,y_indice=next(valid_indice_generator)

        Y=[y1_label[y_indice],y2_label[y_indice]]

        yield X,Y

train_gen=trainGen()

valid_gen=validGen()
#next(train_gen)
from keras.applications import ResNet50

from keras.applications import InceptionV3

from keras.applications import InceptionResNetV2

def build_model1():

    conv=ResNet50(include_top=False,input_shape=(224,224,1),weights=None)

    conv.trainable=True



    input_layer=Input((224,224,1))

    x=conv(input_layer)

    x=GlobalAveragePooling2D()(x)

    base=BatchNormalization()(x)



    x1=Dense(512)(base)

    x1=BatchNormalization()(x1)

    x1=Activation(activation='relu')(x1)

    x1=Dropout(0.5)(x1)

    output_layer1=Dense(1,activation='sigmoid')(x1)



    x2=Dense(512)(base)

    x2=BatchNormalization()(x2)

    x2=Activation(activation='relu')(x2)

    x2=Dropout(0.5)(x2)

    output_layer2=Dense(1,activation='sigmoid')(x2)



    model=models.Model(inputs=input_layer,outputs=[output_layer1,output_layer2])

    model.compile(optimizer=optimizers.Adam(lr=0.001),

                      loss='binary_crossentropy',

                      metrics=['acc'])

    print(model.summary())

    return model



def build_model2():

    conv=InceptionV3(include_top=False,input_shape=(299,299,1),weights=None)

    conv.trainable=True



    input_layer=Input((299,299,1))

    x=conv(input_layer)

    x=GlobalAveragePooling2D()(x)

    base=BatchNormalization()(x)



    x1=Dense(512)(base)

    x1=BatchNormalization()(x1)

    x1=Activation(activation='relu')(x1)

    x1=Dropout(0.5)(x1)

    output_layer1=Dense(1,activation='sigmoid')(x1)



    x2=Dense(512)(base)

    x2=BatchNormalization()(x2)

    x2=Activation(activation='relu')(x2)

    x2=Dropout(0.5)(x2)

    output_layer2=Dense(1,activation='sigmoid')(x2)



    model=models.Model(inputs=input_layer,outputs=[output_layer1,output_layer2])

    model.compile(optimizer=optimizers.Adam(lr=0.001),

                      loss='binary_crossentropy',

                      metrics=['acc'])

    print(model.summary())

    return model



def build_model3():

    conv=InceptionResNetV2(include_top=False,input_shape=(299,299,1),weights=None)

    conv.trainable=True



    input_layer=Input((299,299,1))

    x=conv(input_layer)

    x=GlobalAveragePooling2D()(x)

    base=BatchNormalization()(x)



    x1=Dense(512)(base)

    x1=BatchNormalization()(x1)

    x1=Activation(activation='relu')(x1)

    x1=Dropout(0.5)(x1)

    output_layer1=Dense(1,activation='sigmoid')(x1)



    x2=Dense(512)(base)

    x2=BatchNormalization()(x2)

    x2=Activation(activation='relu')(x2)

    x2=Dropout(0.5)(x2)

    output_layer2=Dense(1,activation='sigmoid')(x2)



    model=models.Model(inputs=input_layer,outputs=[output_layer1,output_layer2])

    model.compile(optimizer=optimizers.Adam(lr=0.001),

                      loss='binary_crossentropy',

                      metrics=['acc'])

    print(model.summary())

    return model
model=build_model3()
reduce=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto')

checkpointer = ModelCheckpoint(filepath='./model_checkpoint_inception_resnet_2.hdf5', verbose=1, save_best_only=True)

earlystopping=EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')

history=model.fit_generator(train_gen,

                            steps_per_epoch=len(train_indice_generator),

                            epochs=200,

                            validation_data=valid_gen,

                            validation_steps=len(valid_indice_generator),

                            callbacks=[reduce,checkpointer,earlystopping])
import matplotlib.pyplot as plt

%matplotlib inline
train_loss=history.history['loss']

train_acc1=history.history['dense_2_acc']

train_acc2=history.history['dense_4_acc']



val_loss=history.history['val_loss']

val_acc1=history.history['val_dense_2_acc']

val_acc2=history.history['val_dense_4_acc']



epochs=range(1,len(train_loss)+1)

plt.plot(epochs,train_loss,'o',label='training loss',color='blue')

plt.plot(epochs,train_acc1,':',label='training accurate 1',color='blue')

plt.plot(epochs,train_acc2,'-',label='training accurate 2',color='blue')



plt.plot(epochs,val_loss,'o',label='validation loss',color='gold')

plt.plot(epochs,val_acc2,':',label='validation accurate 1',color='gold')

plt.plot(epochs,val_acc2,'-',label='validation accurate 2',color='gold')

plt.legend()



plt.show()