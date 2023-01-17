import pandas as pd
import numpy as np
import datetime
import random
import glob
import cv2
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense,concatenate,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100

%matplotlib inline

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# 乱数シード固定
seed_everything(2020)
inputPath = '/kaggle/input/aiacademydeeplearning/'
# 画像読み込み
image = cv2.imread(inputPath+'train_images/1_bathroom.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
display(image.shape)
display(image[0][0])
# 画像を表示
plt.figure(figsize=(8,4))
plt.imshow(image)
# 画像のサイズ変更
image = cv2.resize(image,(256,256))
display(image.shape)
display(image[0][0])
# 画像を表示
plt.figure(figsize=(8,4))
plt.imshow(image)
train = pd.read_csv(inputPath+'train.csv')
test = pd.read_csv(inputPath+'test.csv' )
display(train.shape)
display(train.head())
display(test.head())

def load_images(df,inputPath,size,roomType):
    images = []
    for i in df['id']:
#    for i in range(1,2):
#        basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,roomType)])
        basePath = os.path.sep.join([inputPath, "{}_*".format(i)])
        housePaths = sorted(list(glob.glob(basePath)))
#        print(housePaths)
        
        id_images = []  
        for housePath in housePaths:

            #print(housePath)
            image = cv2.imread(housePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (size, size))
            id_images.append(image)
        h_image1 = cv2.hconcat([id_images[0],id_images[1]])
        h_image2 = cv2.hconcat([id_images[2],id_images[3]])
        merge_image = cv2.vconcat([h_image1,h_image2])
        images.append(merge_image)
        plt.imshow(merge_image)
        
        #plt.imshow(merge_image)
        #images.append(image)
    
    np_images = np.array(images) / 255.0
   
    return np_images






# load train images
inputPath = '/kaggle/input/aiacademydeeplearning/train_images/'
size = 32
roomType = 'kitchen'
train_images = load_images(train,inputPath,size,roomType)
display(train_images.shape)
display(train_images[0][0][0])



# load test images
inputPath = '/kaggle/input/aiacademydeeplearning/test_images/'
size = 32
roomType = 'kitchen'
test_images = load_images(test,inputPath,size,roomType)
display(test_images.shape)
display(test_images[0][0][0])
#8:2
#train_x, valid_x, train_images_x, valid_images_x = train_test_split(train, train_images, test_size=0.2)
#train_y = train_x['price'].values
#valid_y = valid_x['price'].values
#display(train_images_x.shape)
#display(valid_images_x.shape)
#display(train_y.shape)
#display(valid_y.shape)

"""

def create_cnn(inputShape):
    model = Sequential()

    # 演習:kernel_sizeを変更してみてください
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid',
                     activation='relu', kernel_initializer='he_normal', input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid', 
                     activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    #演習:もう一層Conv2D->MaxPooling2D->BatchNormalization->Dropoutを追加してください
    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='valid', 
                     activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    
    model.add(Flatten())
    
    model.add(Dense(units=256, activation='relu',kernel_initializer='he_normal'))  
    model.add(Dense(units=32, activation='relu',kernel_initializer='he_normal'))    
    model.add(Dense(units=1, activation='linear'))
    
    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 
    return model
"""
#def create_cnn(inputShape):
def create_cnn(inputShape,inputShapeTxt):
    
    img_inputs = Input(shape=inputShape, name='img_inputs')

    
    #1
    conv_1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid',activation='relu', kernel_initializer='he_normal')(img_inputs)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    BatchNz_1 = BatchNormalization()(maxpool_1)
    Dropout_1 = Dropout(0.1)(BatchNz_1)
    #2
    conv_2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='he_normal')(Dropout_1)
    maxpool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    BatchNz_2 = BatchNormalization()(maxpool_2)
    Dropout_2 = Dropout(0.1)(BatchNz_2)
    #3
    conv_3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='he_normal')(Dropout_2)
    maxpool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
    BatchNz_3 = BatchNormalization()(maxpool_3)
    Dropout_3 = Dropout(0.1)(BatchNz_3)
    
    img_flatten = Flatten()(Dropout_3)
    
    
    
    txt_imputs =  Input(shape=inputShapeTxt, name='txt_imputs')
    
    
    # merge
    merged = concatenate([img_flatten, txt_imputs],axis=1)
    
    
    
    dense_1 = Dense(units=256, activation='relu',kernel_initializer='he_normal')(merged)
    dense_2 = Dense(units=32, activation='relu',kernel_initializer='he_normal')(dense_1)
    output = Dense(units=1, activation='relu',kernel_initializer='he_normal')(dense_2)
    
    #model = Model(inputs=img_inputs, outputs=output)
    model = Model(inputs=[img_inputs,txt_imputs], outputs=output)
      
    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 
    return model
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


train
train_images

#KFold
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm as tqdm

scores = []

skf = KFold(n_splits=5, random_state=1, shuffle=True)

for i, (train_ix, val_ix) in tqdm(enumerate(skf.split(train, train_images))):
#for train_ix, val_ix in skf.split(train, train_images):
    train_x, train_images_x = train.iloc[train_ix], train_images[train_ix]
    valid_x, valid_images_x = train.iloc[val_ix], train_images[val_ix]
    # https://blog.amedama.jp/entry/2018/06/21/235951 ハマったのでメモ
    
    train_y = train_x['price'].values
    train_x.drop(['price'], axis=1, inplace=True)
    valid_y = valid_x['price'].values
    valid_x.drop(['price'], axis=1, inplace=True)
    #print(train_ix)
    print(train_images_x.shape)
    print(valid_images_x.shape)
    print(train_y.shape)
    print(valid_y.shape)
    train_x_multi = train_x.values
    valid_x_multi = valid_x.values
    print(train_x_multi.shape)
    print(valid_x_multi.shape)
     
    #モデル訓練
    # callback parameter
    filepath = "cnn_best_model.hdf5" 
    es = EarlyStopping(patience=5, mode='min', verbose=1) 
    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=5, verbose=1,  mode='min')
    size2=size*2

    # 訓練実行
    inputShape = (size2, size2, 3)
    inputShapeTxt = (5)
    model = create_cnn(inputShape,inputShapeTxt)

    
    #model.fit(train_images_x, train_y, validation_data=(valid_images_x, valid_y),epochs=50, batch_size=16, callbacks=[es, checkpoint, reduce_lr_loss])
    model.fit([train_images_x,train_x_multi], train_y, validation_data=([valid_images_x,valid_x_multi], valid_y),epochs=50, batch_size=16, callbacks=[es, checkpoint, reduce_lr_loss])
    # load best model weights
    model.load_weights(filepath)

    # 評価
    #valid_pred = model.predict(valid_images_x, batch_size=32).reshape((-1,1))
    valid_pred = model.predict([valid_images_x,valid_x_multi], batch_size=32).reshape((-1,1))
    mape_score = mean_absolute_percentage_error(valid_y, valid_pred)
    #print (mape_score)
    scores.append(mape_score)
    
    #print('CV Score of Fold_%d is %f' % (i, mape_score))
       
model.summary()
plot_model(model, to_file='cnn.png', show_shapes=True)
inputPath = '/kaggle/input/aiacademydeeplearning/'
train = pd.read_csv(inputPath+'train.csv')
train_y_all = train['price'].values
train.drop(columns='price', axis=1, inplace=True)
train_x_multi = train.values

inputShape = (size2, size2, 3)
inputShapeTxt = (5)
#model = create_cnn(inputShape,inputShapeTxt)
model.fit([train_images,train_x_multi], train_y_all,epochs=50, batch_size=16, callbacks=[es, checkpoint, reduce_lr_loss])

## callback parameter
#filepath = "cnn_best_model.hdf5" 
#es = EarlyStopping(patience=5, mode='min', verbose=1) 
#checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 
#reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=5, verbose=1,  mode='min')
#size2=size*2

## 訓練実行
#inputShape = (size2, size2, 3)
#model = create_cnn(inputShape)
#model.fit(train_images_x, train_y, validation_data=(valid_images_x, valid_y),epochs=50, batch_size=16,
#    callbacks=[es, checkpoint, reduce_lr_loss])
#def mean_absolute_percentage_error(y_true, y_pred): 
#    y_true, y_pred = np.array(y_true), np.array(y_pred)
#    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#
## load best model weights
#model.load_weights(filepath)
#
## 評価
#valid_pred = model.predict(valid_images_x, batch_size=32).reshape((-1,1))
#mape_score = mean_absolute_percentage_error(valid_y, valid_pred)
#print (mape_score)
plot_model(model, to_file='cnn.png')
#test_pred = model.predict(test_images, batch_size=32).reshape((-1,1))
test_txt_multi=test.values
test_pred = model.predict([test_images,test_txt_multi], batch_size=32).reshape((-1,1))

df_test_pred = pd.DataFrame(np.round(test_pred))
submission = test

submission['price'] = df_test_pred
submission.drop(['bedrooms','bathrooms','area','zipcode'], axis=1, inplace=True)
print(submission)

submission.to_csv('submission.csv',index=False)