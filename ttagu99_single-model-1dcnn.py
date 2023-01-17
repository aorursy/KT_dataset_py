# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import tensorflow as tf

print(tf.__version__)

strategy  = tf.distribute.MirroredStrategy()# ['/gpu:0']) 

data_root="../"
print(os.listdir(data_root+ "input/ds2-ds5-competition-1/"))
train_org = pd.read_csv(data_root+ "input/ds2-ds5-competition-1/train.csv")

test_org = pd.read_csv(data_root+ "input/ds2-ds5-competition-1/test.csv")

submission = pd.read_csv(data_root+ "input/ds2-ds5-competition-1/sample_submission.csv")
train_org.info()
neg_ratio = train_org[train_org.label == 0].shape[0] / train_org.shape[0]

print(neg_ratio)
train_org['group']=0

garr = np.empty(len(train_org))

garr
pre_label = 0.0

group = 0

min_signal_len=float('inf')

cnt=0

for idx, label in enumerate(train_org.label):

    if pre_label != label:

        if cnt<min_signal_len:

            min_signal_len=cnt

        cnt=0

        group +=1

        pre_label = label

    garr[idx]=group

    cnt+=1

    

train_org['group'] = garr



print('label signal min length:',min_signal_len)
from collections import Counter

group_c = Counter(garr)
train_org.head()
X = train_org.copy()

x_cols =['s'+ str(i) for i in list(range(1,17,1))]

X = X[x_cols]

X.head(10)
print('train_org.shape:',train_org.shape)

print('test_org.shape:',test_org.shape)
train_org.describe()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_score

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from sklearn.decomposition import PCA

from sklearn.cross_decomposition import PLSRegression, PLSSVD

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer, StandardScaler, Normalizer, MinMaxScaler

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.svm import SVC

#import lightgbm as lgb

import random

import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv1D, AvgPool1D, MaxPooling1D, BatchNormalization, Activation,GlobalAveragePooling1D, GlobalMaxPool1D, concatenate, Dense, Dropout

from tensorflow.keras.optimizers import RMSprop, SGD

from tensorflow.keras import layers

from tensorflow.keras.layers import Dense, Dropout, Reshape, Conv1D,  Activation, AveragePooling1D, GlobalAveragePooling1D, Lambda, Input, Concatenate, Add, UpSampling1D, Multiply

from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from tensorflow.keras.callbacks import Callback

from tensorflow.keras.losses import mean_squared_error, mean_absolute_error

from sklearn.metrics import cohen_kappa_score, f1_score

from sklearn.model_selection import KFold, train_test_split

from tensorflow.keras.utils import Sequence

# from tensorflow_addons.optimizers import SWA  

# 이 노트북에서는 사용 안했음, 1개 fold set 만 학습했음.시간때문에 미미한 score증가 있음 



def se_block(x_in, layer_n, stage, block):

    se_name_base = 'se' + str(stage) + block

    x = Conv1D(11, (1), strides=1,name=se_name_base + 'conv')(x_in)

    x = GlobalAveragePooling1D(name=se_name_base + "gap")(x_in)

    x = Dense(layer_n, activation="sigmoid", name=se_name_base + "se2")(x)

    x_out=Multiply( name=se_name_base + "mul")([x_in, x])

    return x_out





def identity_block_1d(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res_1d' + str(stage) + block + '_branch'

    bn_name_base = 'bn_1d' + str(stage) + block + '_branch'

    x = Conv1D(filters1, (1), name=conv_name_base + '2a')(input_tensor)

    x = BatchNormalization( name=bn_name_base + '2a')(x)

    x = Activation('relu')(x)

    x = Conv1D(filters2, kernel_size,

               padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)

    x = Activation('relu')(x)

    x = Conv1D(filters3, (1), name=conv_name_base + '2c')(x)

    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])

    x = Activation('relu')(x)

    return x



def conv_block_1d(input_tensor, kernel_size, filters, stage, block, strides=(1), pool_size=2, use_se= True, use_att=True):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res_1d' + str(stage) + block + '_branch'

    bn_name_base = 'bn_1d' + str(stage) + block + '_branch'

    x = Conv1D(filters1, (1), strides=strides,

               name=conv_name_base + '2a')(input_tensor)

    x = BatchNormalization(name=bn_name_base + '2a')(x)

    x = Activation('relu')(x)

    x = Conv1D(filters2, kernel_size, padding='same',

               name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)

    x = Activation('relu')(x)

    shortcut = Conv1D(filters2, (1), strides=strides,

                      name=conv_name_base + '1')(input_tensor)

    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])

    x = Activation('relu')(x)

    

    x = MaxPooling1D((pool_size), padding='same', name=conv_name_base + '_max_pool')(x)

    x = Conv1D(filters3, (1), name=conv_name_base + '2c')(x)

    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = Activation('relu')(x)

    

    if use_se == True:

        x = se_block(x, filters3, stage, block)



    return x



def up_conv_block_1d(input_tensor, kernel_size, filters, stage, block, strides=(1), pool_size=2):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res_1d' + str(stage) + block + '_branch'

    bn_name_base = 'bn_1d' + str(stage) + block + '_branch'

    x = UpSampling1D(pool_size)(input_tensor)

    x = Conv1D(filters1, (1), name=conv_name_base + '2a')(x)

    x = BatchNormalization(name=bn_name_base + '2a')(x)

    x = Activation('relu')(x)

    x = Conv1D(filters2, kernel_size, padding='same',

               name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)

    x = Activation('relu')(x)

    x = Conv1D(filters3, (1), name=conv_name_base + '2c')(x)

    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv1D(filters3, (1), strides=strides,

                      name=conv_name_base + '1')(input_tensor)

    

    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = Activation('relu')(x)

    return x





atrain = train_org.copy()

# atrain['split'] = (train.time/1000).astype(np.int)%5





mmScaler = MinMaxScaler()

mmScaler.fit(X)

#mmsX = mmScaler.transform(X)
def get_model( features=16, layer_n_1st=16, kernel_size=5, input_len=1600 , pools=[2,2,2], strides=[2,1,1]):

    filters= np.array([layer_n_1st, layer_n_1st*2, layer_n_1st*4])

    with strategy.scope():

        input_layer = Input(shape=(input_len,features))

        x = input_layer

        

        for idx in range(len(pools)):  

            x = conv_block_1d(x,kernel_size, filters, idx, 'conv',pool_size=pools[idx], strides=strides[idx])

            filters *=2



        x = GlobalAveragePooling1D()(x)

        reg = Dense(1, activation= 'linear', name='reg')(x)

        cls = Dense(1,activation='sigmoid', name='cls')(x)

        model = Model(inputs = input_layer, outputs = [reg,cls])

        return model
model=get_model()

model.summary()
# batch_size=512

# crop_size=6401

# pools=[2,2,2,2,2]

# strides=[2,1,1,1,1]

# layer_n_1st = 16

# kernel_size=3

# features=16

# use_aug=False

# sel_valnumber =3 #0~4

# tr_idx = [0,1,2,4,5]



# atrain['split'] = train_org.group%6

# ho_df = atrain[atrain['split']==3]  # 나중에 ensemble, stacking model 학습을 위해 빼놓은 set

# train_val_df = atrain[atrain['split']!=3]



# train_df = train_val_df[train_val_df['split']!=tr_idx[sel_valnumber]]

# val_df = train_val_df[train_val_df['split']==tr_idx[sel_valnumber]]

# print(train_df.shape, val_df.shape, ho_df.shape)

# train_s = mmScaler.transform(train_df[x_cols])

# val_s = mmScaler.transform(val_df[x_cols])

# test_s = mmScaler.transform(test_org[x_cols])

# print(train_s.shape, val_s.shape, test_s.shape)
class gas_generator(Sequence):

    def __init__(self, data, batch_size,target=None, crop_size = 1601, state='Train', use_aug=False):

        self.data = data

        if state!='Test':

            self.target = target#enc.transform(target)    

        self.batch_size = batch_size

        self.crop_size = crop_size

        self.state = state

        self.n = 0

        self.use_aug = use_aug

        self.on_epoch_end()  

        

    def __len__(self):

        return self.len

    

    def __getitem__(self, index):

        batch_idx = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        b_data = np.empty((self.batch_size, self.crop_size, self.data.shape[1]), dtype = float)

        if self.state!='Test':

            b_target = np.empty((self.batch_size, 1), dtype = float)

            b_target_cls = np.empty((self.batch_size, 1), dtype = float)

        for i, k in enumerate(batch_idx):

            wav = self.data[k-self.crop_size//2:k+self.crop_size//2+1]

#             if self.use_aug ==True and np.random.rand()<0.5:

#            wav = np.flipud(wav)



            b_data[i] = wav

            if self.state!='Test':

                tar = self.target.iloc[k] #tf.keras.utils.to_categorical(self.target[k:k+self.crop_size]

                                                    #,num_classes=len(enc.classes_)) 

                b_target[i] = np.expand_dims(tar,axis=-1)#tar #np.expand_dims(tar,axis=-1)

                b_target_cls[i] = np.expand_dims(tar>0,axis=-1)

        if self.state!='Test':

            return b_data, [b_target, b_target_cls]

        else:

            return b_data

        

    def __next__(self):

        if self.n >= self.len:

            self.n = 0

        result = self.__getitem__(self.n)

        self.n += 1

        return result

        

    def get_data_len(self):

        return len(self.indices)



    def get_indices(self):

        return self.indices



    def on_epoch_end(self):

        if self.state == 'Train':

            self.indices = np.random.randint(self.crop_size//2, len(self.data)-self.crop_size//2, size=len(self.data)-self.crop_size)

        else:

            self.indices = np.arange(self.crop_size//2, len(self.data)-self.crop_size//2)

        self.len = -(-len(self.indices)//self.batch_size)
# ds = tf.data.Dataset.from_generator(

#     gas_generator, args=[val_s,target=val_df.label,batch_size=batch_size,crop_size=crop_size], 

#     output_types=(tf.float32, (tf.float32,tf.float32)), 

#     output_shapes=([None,6401,16], ([None,1],[None,1]))

# )
batch_size=512

crop_size=6401

pools=[2,2,2,2,2]

strides=[2,1,1,1,1]

layer_n_1st = 16

kernel_size=3

features=16

use_aug=False

sel_valnumber =3 #0~4

tr_idx = [0,1,2,4,5]



atrain['split'] = train_org.group%6

ho_df = atrain[atrain['split']==3]  # 나중에 ensemble, stacking model 학습을 위해 빼놓은 set

train_val_df = atrain[atrain['split']!=3]



train_df = train_val_df[train_val_df['split']!=tr_idx[sel_valnumber]]

val_df = train_val_df[train_val_df['split']==tr_idx[sel_valnumber]]

print(train_df.shape, val_df.shape, ho_df.shape)

train_s = mmScaler.transform(train_df[x_cols])

val_s = mmScaler.transform(val_df[x_cols])

test_s = mmScaler.transform(test_org[x_cols])

print(train_s.shape, val_s.shape, test_s.shape)



train_gen = gas_generator(train_s,target=train_df.label,batch_size=batch_size,crop_size=crop_size,use_aug=use_aug)

val_gen = gas_generator(val_s,target=val_df.label,batch_size=batch_size,crop_size=crop_size)





best_model_path = 'best_1dcnn_reg' +str(crop_size) +str(batch_size) +str(pools)+str(strides)

best_model_path+=str(features)+str(kernel_size) + str(layer_n_1st) +str(sel_valnumber)#+str(use_aug)

check_point=tf.keras.callbacks.ModelCheckpoint(monitor='val_reg_loss',verbose=1

    ,filepath=best_model_path+'.h5',save_best_only=True,mode='min') 

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_reg_loss', factor=0.1

            ,verbose=1, patience=3, min_lr=0.00001,min_delta=0.00001) 

csvlogger=tf.keras.callbacks.CSVLogger(filename=best_model_path+".log", append=False)



losses = {'cls':binary_crossentropy,'reg':mean_absolute_error}

loss_weights={'cls':0.5,'reg':0.5}

met= {'cls':'accuracy'}

with strategy.scope():

    model = get_model(features =features,layer_n_1st=layer_n_1st, 

                      kernel_size=kernel_size, input_len=crop_size, pools=pools, strides=strides)

    

    model.compile(loss=losses, 

                  optimizer=Adam(lr=0.001)

                  ,metrics=met

                    ,loss_weights = loss_weights

                 )

    

hist = model.fit(

    train_gen,# steps_per_epoch=10, validation_steps=10,

    epochs = 1, # 캐글 노트북 지속 실행시간이 최대 9시간으로 epoch을 줄였음, 원래는 15 epoch

    validation_data=val_gen,

    workers=1,

    verbose = 1

    ,callbacks=[check_point,reduce_lr,csvlogger]

    )



# Stocastic Weight Averaging~~ 이 노트북에서는 사용 안함.



# print(best_model_path)

# if os.path.isfile(best_model_path+'.h5'):

#     print('load weight re start train')

#     model.load_weights(best_model_path+'.h5')



# with strategy.scope():

#     model.compile(loss=losses, 

#                   optimizer=SWA(Adam(lr=0.0001))

#                   ,metrics=met

#                     ,loss_weights = loss_weights

#                  )

    

# swa_model_path = best_model_path + '_swa'

# check_point=tf.keras.callbacks.ModelCheckpoint(monitor='val_reg_loss',verbose=1

#     ,filepath=swa_model_path+'.h5',save_best_only=True,mode='min') 

# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_reg_loss', factor=0.1

#             ,verbose=1, patience=3, min_lr=0.00001,min_delta=0.00001) 

# csvlogger=tf.keras.callbacks.CSVLogger(filename=swa_model_path+".log", append=False)

# swa_hist = model.fit(

#     train_gen,

#     epochs = 10,

#     validation_data=val_gen,

#     workers=1,

#     verbose = 1

#     ,callbacks=[check_point,reduce_lr,csvlogger]

#     )
print(best_model_path+'.h5')

model.load_weights(best_model_path+'.h5')
def cut_value(cut_new_y):

    for idx, v in enumerate(cut_new_y):

        if v<0:

            cut_new_y[idx]=0.0

        if v>533.33:

            cut_new_y[idx]=533.33

    return cut_new_y
val_s.shape

val_df.shape
val_gen = gas_generator(val_s,batch_size=batch_size,crop_size=crop_size,state='Test')

val_prd = model.predict(val_gen,verbose=1)
idx_val = val_gen.get_indices()

cut_new_y_val=np.zeros(len(val_df))

cut_new_y_val[idx_val] = (val_prd[0]*(val_prd[1]>0.5))[:len(idx_val)].squeeze()

cut_new_y_val = cut_value(cut_new_y_val)



from sklearn.metrics import mean_absolute_error as skmae

score = skmae(val_df.label,cut_new_y_val)

print(f'val score : {score:.4f}')

ho_s = mmScaler.transform(ho_df[x_cols])

ho_gen = gas_generator(ho_s,batch_size=batch_size,crop_size=crop_size,state='Test')

ho_prd = model.predict(ho_gen,verbose=1)

idx_ho = ho_gen.get_indices()

cut_new_y_ho=np.zeros(len(ho_df))

cut_new_y_ho[idx_ho] = (ho_prd[0]*(ho_prd[1]>0.5))[:len(idx_ho)].squeeze()

cut_new_y_ho = cut_value(cut_new_y_ho)



from sklearn.metrics import mean_absolute_error as skmae

score = skmae(ho_df.label,cut_new_y_ho)

print(f'ho score : {score:.4f}')

np.save(best_model_path+'_ho.npy', cut_new_y_ho)

test_gen = gas_generator(test_s,batch_size=batch_size,crop_size=crop_size,state='Test')

test_prd = model.predict(test_gen,verbose=1)

idx = test_gen.get_indices()

cut_new_y=np.zeros(len(submission))

cut_new_y[idx] = (test_prd[0]*(test_prd[1]>0.5))[:len(idx)].squeeze()

cut_new_y = cut_value(cut_new_y)

np.save(best_model_path+'_test.npy', cut_new_y)
submission_cnn = submission.copy()

submission_cnn['label'] = cut_new_y
submission_cnn.to_csv(best_model_path+f'submission{score:.4f}.csv', index=False)