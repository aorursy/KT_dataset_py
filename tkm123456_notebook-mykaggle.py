# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.endswith('.jpg'):

            break

        print(os.path.join(dirname, filename))

        

df = pd.read_csv('../input/herbarium-2020-fgvc7/sample_submission.csv')

import json,codecs

with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/train/metadata.json", 'r',

                 encoding='utf-8', errors='ignore') as f:

    train_meta = json.load(f)

    

with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/test/metadata.json", 'r',

                 encoding='utf-8', errors='ignore') as f:

    test_meta = json.load(f)



train_df = pd.DataFrame(train_meta['annotations'])

train_cat = pd.DataFrame(train_meta['categories'])

train_cat.columns =['family','genus','category_id','category_name']

train_img = pd.DataFrame(train_meta['images'])

train_img.columns = ['file_name', 'height', 'image_id', 'license', 'width']

train_reg = pd.DataFrame(train_meta['regions'])

train_reg.columns = ['region_id', 'region_name']

train_df = train_df.merge(train_cat, on='category_id', how='outer')

train_df = train_df.merge(train_img, on='image_id', how='outer')

train_df = train_df.merge(train_reg, on='region_id', how='outer')

na = train_df.file_name.isna()

#display(na) #各行をチェックしNaNならTrueを返す



keep = [x for x in range(train_df.shape[0]) if not na[x]] 

train_df = train_df.iloc[keep]



dtypes = ['int32', 'int32', 'int32', 'int32', 'object', 'object', 'object', 'object', 'int32', 'int32', 'int32', 'object']

for n, col in enumerate(train_df.columns): #n -> 行数 col -> Index

    train_df[col] = train_df[col].astype(dtypes[n]) #型の変換

    

test_df = pd.DataFrame(test_meta['images'])

test_df.columns = ['file_name', 'height', 'image_id', 'license', 'width']



train_df.to_csv('full_train_data.csv', index=False)

test_df.to_csv('full_test_data.csv', index=False)



# print("Total Unique Values for each columns:")

# print("{0:10s} \t {1:10d}".format('train_df', len(train_df)))



#Data Explpration



for col in train_df.columns:

    print("{0:10s} \t {1:10d}".format(col, len(train_df[col].unique())))



family = train_df[['family', 'genus', 'category_name']].groupby(['family', 'genus']).count()



print("Sequence End")

print(train_df)
import tensorflow.keras as keras

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization, Input, concatenate,add,Add

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split as tts



in_out_size = (120*120) + 3 #We will resize the image to 120*120 and we have 3 outputs

def xavier(shape, dtype=None):

    return np.random.rand(*shape)*np.sqrt(1/in_out_size)
"""

## Model function

def fg_model(shape,lr=0.001):

    i = Input(shape)

    

    x = Conv2D(3,(3,3),activation='relu',padding='same',kernel_initializer=xavier)(i)

    x = Conv2D(3,(5,5),activation='relu',padding='same',kernel_initializer=xavier)(x)

    x = MaxPool2D(pool_size=(3,3),strides=(3,3))(x)

    x = Dropout(0.5)(x)

    x = Conv2D(16,(5,5),activation='relu',padding='same',kernel_initializer=xavier)(x)

    x = MaxPool2D(pool_size=(5,5),strides=(5,5))(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Flatten()(x)

    

    o1 = Dense(310,activation='softmax',name='family',kernel_initializer=xavier)(x)

    

    o2 = concatenate([o1,x])

    o2 = Dense(3678,activation='softmax',name='genus',kernel_initializer=xavier)(o2)

    

    o3 = concatenate([o1,o2,x])

    o3 = Dense(32094,activation='softmax',name='category_id',kernel_initializer=xavier)(o3)

    

    x = Model(inputs=i, outputs=[o1,o2,o3])

    

    opt = Adam(lr=lr,amsgrad=True)

    x.compile(optimizer=opt,loss=['sparse_categorical_crossentropy', 

                                   'sparse_categorical_crossentropy', 

                                   'sparse_categorical_crossentropy'],

                 metrics=['accuracy'])

    

    return x



model = fg_model((120,120,3))

model.summary()

plot_model(model, to_file='full_model_plot.png',show_shapes=True,show_layer_names=True)

"""
"""

function imformation

name:fg_model

argment:

1: shape -> (120,120,3)

2: lr -> learning rate 0.001 is default value



This model was based on dl4us lesson2_sec4

"""

def fg_model(shape,lr=0.001):

    i = Input(shape)



    x = (Conv2D(5, kernel_size=(5, 5), activation='relu',padding='same',

                     kernel_initializer='he_normal'))(i) #120*120*3 -> 120*120*5

    x = (MaxPool2D(pool_size=(2, 2)))(x) # 120*120*5 -> 60*60*5

    x = (Conv2D(5, kernel_size=(3, 3), activation='relu',padding='same',

                     kernel_initializer='he_normal'))(x) # 60*60*5 -> 60*60*5

    x = (MaxPool2D(pool_size=(2, 2)))(x) #60*60*5 -> 30*30*5

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = (Flatten())(x) #30*30*5 -> 4500

    #x = (Dense(600, activation='relu', 

    #                kernel_initializer='he_normal'))(x)

    #x = (Dense(32094, activation='softmax'))(x) # category_name 	      32093



    o1 = Dense(310,activation='softmax',name='family',kernel_initializer='he_normal')(x)

    o2 = Dense(3678, activation='softmax', name='genus', kernel_initializer='he_normal')(x)

    o3 = Dense(32094, activation='softmax',name='category_id', kernel_initializer='he_normal')(x)

    y = Model(inputs=i,outputs=[o1,o2,o3])

    

    y.compile(

        #loss=keras.losses.sparse_categorical_crossentropy,

        loss=['sparse_categorical_crossentropy', 

               'sparse_categorical_crossentropy', 

               'sparse_categorical_crossentropy'],

        optimizer='adam',

        metrics=['accuracy']

    )

              

    return y

              

model = fg_model((120,120,3))

model.summary()

plot_model(model, to_file='full_model_plot.png',show_shapes=True,show_layer_names=True)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(featurewise_center=False,

                                      featurewise_std_normalization=False,

                                      rotation_range=180,

                                      width_shift_range=0.1,

                                      height_shift_range=0.1,

                                      zoom_range=0.2)



"""

ImageDataGenerator -> 画像データの角度を変えたり、移動・回転させたりすることのできる亜種画像が生成できる





"""

display(datagen)
m = train_df[['file_name','family','genus','category_id']]

display(m)
#m = train_df[['file_name','family','genus','category_id']]

## family genusが文字列となっているので数値に置き換える

fam = m.family.unique().tolist()

m.family = m.family.map(lambda x:fam.index(x))

gen = m.genus.unique().tolist()

m.genus = m.genus.map(lambda x:gen.index(x))

display(m)
"""

train,verif = tts(m,test_size=0.2,shuffle=True,random_state=17)

train = train[:40000]

verif = verif[:10000]

shape = (120,120,3)

epochs = 2

batch_size = 32



model = fg_model(shape,0.007)



for layers in model.layers:

    if layers.name == 'genus' or layers.name == 'category_id':

        layers.trainable = False



#Train Family for 2 epochs

model.fit_generator(train_datagen.flow_from_dataframe(dataframe=train,

                                                      directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                                                      x_col="file_name",

                                                      y_col=["family", "genus", "category_id"],

                                                      target_size=(120, 120),

                                                      batch_size=batch_size,

                                                      class_mode='multi_output'),

                    validation_data=train_datagen.flow_from_dataframe(

                        dataframe=verif,

                        directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                        x_col="file_name",

                        y_col=["family", "genus", "category_id"],

                        target_size=(120, 120),

                        batch_size=batch_size,

                        class_mode='multi_output'),

                    epochs=epochs,

                    steps_per_epoch=len(train)//batch_size,

                    validation_steps=len(verif)//batch_size,

                    verbose=1,

                    workers=8,

                    use_multiprocessing=False)



#Reshuffle the inputs

train, verif = tts(m, test_size=0.2, shuffle=True, random_state=17)

train = train[:40000]

verif = verif[:10000]



#Make the Genus layer Trainable

for layers in model.layers:

    if layers.name == 'genus':

        layers.trainable = True

        

#Train Family and Genus for 2 epochs

model.fit_generator(train_datagen.flow_from_dataframe(dataframe=train,

                                                      directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                                                      x_col="file_name",

                                                      y_col=["family", "genus", "category_id"],

                                                      target_size=(120, 120),

                                                      batch_size=batch_size,

                                                      class_mode='multi_output'),

                    validation_data=train_datagen.flow_from_dataframe(

                        dataframe=verif,

                        directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                        x_col="file_name",

                        y_col=["family", "genus", "category_id"],

                        target_size=(120, 120),

                        batch_size=batch_size,

                        class_mode='multi_output'),

                    epochs=epochs,

                    steps_per_epoch=len(train)//batch_size,

                    validation_steps=len(verif)//batch_size,

                    verbose=1,

                    workers=8,

                    use_multiprocessing=False)



#Reshuffle the inputs

train, verif = tts(m, test_size=0.2, shuffle=True, random_state=17)

train = train[:40000]

verif = verif[:10000]



#Make the category_id layer Trainable

for layers in model.layers:

    if layers.name == 'category_id':

        layers.trainable = True

        

#Train them all for 2 epochs

model.fit_generator(train_datagen.flow_from_dataframe(dataframe=train,

                                                      directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                                                      x_col="file_name",

                                                      y_col=["family", "genus", "category_id"],

                                                      target_size=(120, 120),

                                                      batch_size=batch_size,

                                                      class_mode='multi_output'),

                    validation_data=train_datagen.flow_from_dataframe(

                        dataframe=verif,

                        directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                        x_col="file_name",

                        y_col=["family", "genus", "category_id"],

                        target_size=(120, 120),

                        batch_size=batch_size,

                        class_mode='multi_output'),

                    epochs=epochs,

                    steps_per_epoch=len(train)//batch_size,

                    validation_steps=len(verif)//batch_size,

                    verbose=1,

                    workers=8,

                    use_multiprocessing=False)

"""

# 一回分けてみてもいいかもしれない

# fit_generatorの引数をそれぞれ説明する

# fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

# generator -> バッチ毎に生成されたデータ(訓練用のデータとして使用)

# validation_data -> 検証用データ

# Train, Testようにそれぞれデータの用意

train,test = tts(m,test_size=0.2,shuffle=True,random_state=17)

train = train[:40000]

test = test[:10000]

shape = (120,120,3)

epochs = 2

batch_size = 32



model = fg_model(shape,0.001)



for layer in model.layers:

    layer.trainable = True



# model.fit_generator(datagen.flow(x_train, y_train, batch_size=100),

#                     steps_per_epoch=x_train.shape[0] // 100, epochs=30, validation_data=(x_valid, y_valid))



# refarence URL about keras flow_from_dataframe

# https://keras.io/ja/preprocessing/image/



# refarence URL about validation_data

# https://keras.io/ja/models/model/

data_generator = datagen.flow_from_dataframe(

                        dataframe=train,

                        directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                        x_col="file_name",

                        y_col=["family", "genus", "category_id"],

                        target_size=(120,120),

                        batch_size=batch_size,

                        class_mode='multi_output') #default

                                                



data_validation = datagen.flow_from_dataframe(

                        dataframe=test,

                        directory='../input/herbarium-2020-fgvc7/nybg2020/train/',

                        x_col="file_name",

                        y_col=["family", "genus", "category_id"],

                        target_size=(120, 120),

                        batch_size=batch_size,

                        class_mode='multi_output')
model.fit_generator(generator=data_generator,validation_data=data_validation,steps_per_epoch=len(train)//batch_size,

                       epochs=epochs,validation_steps=len(test)//batch_size,verbose=1,use_multiprocessing=False)
# モデルの評価

score = model.evaluate(test.x_col, test.y_col,verbose=0)

print('Test Data loss:', score[0])

print('Test Data accuracy:', score[1])
# This is same as refarence code

batch_size = 32

test_datagen = ImageDataGenerator(featurewise_center=False,

                                  featurewise_std_normalization=False)



generator = test_datagen.flow_from_dataframe(

        dataframe = test_df.iloc[:10000], #Limiting the test to the first 10,000 items

        directory = '../input/herbarium-2020-fgvc7/nybg2020/test/',

        x_col = 'file_name',

        target_size=(120, 120),

        batch_size=batch_size,

        class_mode=None,  # only data, no labels

        shuffle=False)



family, genus, category = model.predict_generator(generator, verbose=1)