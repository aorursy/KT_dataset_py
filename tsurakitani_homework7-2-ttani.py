import pandas as pd

import numpy as np

import datetime

import random

import glob

import cv2

import os

import sys

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from category_encoders import OrdinalEncoder, OneHotEncoder

import tensorflow as tf

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense, Input, concatenate

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

import matplotlib.pyplot as plt

import itertools

import datetime

%matplotlib inline



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



seed_everything(2020)

# テーブルデータ読み込み

INPUT_DIR = '../input/4th-datarobot-ai-academy-deep-learning/'



df_train = pd.read_csv(INPUT_DIR + 'train.csv')

print(df_train.shape)



df_pred = pd.read_csv(INPUT_DIR + 'test.csv')

print(df_pred.shape)

# 画像データ読み込み

TRAIN_IMG_DIR = INPUT_DIR + 'images/train_images/'

TEST_IMG_DIR = INPUT_DIR + 'images/test_images/'



# サブミットするときはサイズ大きくする?

IMG_SIZE = 64



def load_img(path):

    # print(path)

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

    return img



def create_miximg(i, img_dir):

    img_kitchen = load_img(img_dir + str(int(i)) + '_kitchen.jpg')

    img_bathroom = load_img(img_dir + str(int(i)) + '_bathroom.jpg')

    img_frontal = load_img(img_dir + str(int(i)) + '_frontal.jpg')

    img_bedroom = load_img(img_dir + str(int(i)) + '_bedroom.jpg')

    img_mix = cv2.hconcat([

        cv2.vconcat([img_kitchen, img_bathroom]),

        cv2.vconcat([img_frontal, img_bedroom])

    ])

    return img_mix



def get_images(df, img_dir):

    images = []

    for i in df['id']:

        images.append(create_miximg(i, img_dir))

    

    # 試しに画像を表示

    plt.figure(figsize=(8,4))

    plt.imshow(images[0])

    

    return np.array(images) / 255.0



train_images = get_images(df_train, TRAIN_IMG_DIR)

print(train_images.shape)



pred_images = get_images(df_pred, TEST_IMG_DIR)

print(pred_images.shape)

# VGGモデル作成

def create_vgg_model(inputShape):

    backbone = VGG16(

        weights='imagenet',

        include_top=False,

        input_shape=inputShape

    )

    for layer in backbone.layers[:5]:

        layer.trainable = False

    #for layer in backbone.layers:

    #    print("{}: {}".format(layer, layer.trainable))

        

    model = Sequential(layers=backbone.layers)

    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=512, activation='relu',kernel_initializer='he_normal'))  

    model.add(Dense(units=32, activation='relu',kernel_initializer='he_normal'))

    

    model.add(Flatten())

    

    return model
# MLPモデル作成

def create_mlp_model(inputShape):

    model = Sequential()

    

    model.add(Dense(units=256, input_shape=inputShape, kernel_initializer='he_normal', activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(units=32,  kernel_initializer='he_normal',activation='relu'))

    model.add(Dropout(0.1))

    

    model.add(Flatten())

    

    return model
# VGGとMLPの結合モデル作成

def create_model(inputShape_vgg, inputShape_mlp):

    

    ## VGGモデル

    vgg_model = create_vgg_model(inputShape_vgg)

    vgg_encoded = vgg_model.output

    

    # MLPモデル

    mlp_model = create_mlp_model(inputShape_mlp)

    mlp_encoded = mlp_model.output

    

    # VGGとMLPの結合

    merged = concatenate([vgg_encoded, mlp_encoded])

    

    # unit数とかDropoutとか調整

    merged = Dense(units=512, kernel_initializer='he_normal', activation='relu')(merged)

    merged = Dropout(0.3)(merged)

    merged = Dense(units=32, kernel_initializer='he_normal', activation='relu')(merged)

    merged = Dropout(0.1)(merged)

    merged = Dense(units=1, activation='linear')(merged)

    

    # 学習プロセスの設定

    model = Model(inputs=[vgg_model.input, mlp_model.input], outputs=merged)

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 

    model.summary()

    

    return model

# MAPE計算(演習そのままだとなんかおかしかったのでちょい修正）

def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(pd.DataFrame(y_pred, columns={'pred'})['pred'])

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# ImageDataGeneratorで画像生成

def gen_flow_inputs(x_image):

    # 生成元の画像がわかるようにインデクスも返す

    x_index = np.arange(x_image.shape[0])

    

    datagen = ImageDataGenerator(

        horizontal_flip=True,

        vertical_flip=True,

        rotation_range=90,

        width_shift_range=0.2,

        height_shift_range=0.2,

    )

    batch = datagen.flow(

        x_image, 

        x_index, 

        batch_size=32, 

        shuffle=True

    )

    batches = 0

    return_x_image = np.copy(x_image)

    return_x_index = np.copy(x_index)

    

    for batch_image, batch_index in batch:

        return_x_image = np.append(return_x_image, batch_image, axis=0)

        return_x_index = np.append(return_x_index, batch_index)

        

        batches += 1

        if batches > 64 : break

    

    return return_x_image, return_x_index



# 特徴量エンジニアリング



df_train_copy = df_train.copy()

df_pred_copy = df_pred.copy()



# trainとtestを区別できるようにフラグを追加してから連結

df_train_copy['data_type'] = 0

df_pred_copy['data_type'] = 1

df_copy = pd.concat([df_train_copy, df_pred_copy])



# 郵便番号は下3桁を切り捨ててOne-Hotにしてみる

df_copy['zipcode'] = (df_copy['zipcode'] // 1000).astype(int)

ohe = OneHotEncoder(cols=['zipcode'],handle_unknown='impute')

df_copy = ohe.fit_transform(df_copy)



# bedrooms, bathroomsは2乗したものと掛けたものを追加

df_copy['bedrooms_2'] = df_copy['bedrooms'] * df_copy['bedrooms']

df_copy['bathrooms_2'] = df_copy['bathrooms'] * df_copy['bathrooms']

df_copy['bedrooms_bathrooms'] = df_copy['bedrooms'] * df_copy['bathrooms']



# areaは標準化

ss = StandardScaler()

df_copy['area'] = ss.fit_transform(np.array(df_copy['area'].values).reshape(-1,1))



# idとpriceを削除

df_copy = df_copy.drop(['id', 'price'], axis=1)



# データを分割する

df_train_copy = df_copy[df_copy.data_type==0].reset_index(drop=True)

df_pred_copy = df_copy[df_copy.data_type==1].reset_index(drop=True)



# print(df_train_copy)

# print(df_pred_copy)

print(df_train_copy.shape)

print(df_pred_copy.shape)
# ImageDataGeneratorで画像と対応するテーブルデータを水増やしする



train_images, x_index = gen_flow_inputs(train_images)

x_train = df_train_copy.iloc[x_index].reset_index(drop=True)

y_train = (df_train['price'].values)[x_index]



print(train_images.shape)

print(type(train_images))



print(x_train.shape)

print(type(x_train))



print(y_train.shape)

print(type(y_train))
# 学習、予測時の各種パラメータ調整

ES_PATIENCE = 5 

PREDICT_BATCH = 32 

EPOCHS = 64 

BATCH_SIZE = 16 

RLR_PATIENCE = 3 

RLR_FACTOR = 0.03



LOOP_NUM = 5


scores = []



#skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

#for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(train_image_vgg, df_train['price'].values))):



for i in range(0, LOOP_NUM):

    print('★検定 ' + str(i) + ' 回目')

    

    # ランダムにデータ分割

    _x_train, _x_valid, _x_train_image, _x_valid_image, _y_train, _y_valid = train_test_split(

        x_train, 

        train_images,

        y_train, 

        test_size=0.2, 

        random_state=i*100

    )

    

    # モデル作成

    inputShape_vgg = _x_train_image.shape[1:]

    inputShape_mlp = (len(_x_train.columns.tolist()), )

    model = create_model(inputShape_vgg, inputShape_mlp)

    

    # コールバック関数

    model_file = 'best_model_' + str(i) + '.hdf5'

    es = EarlyStopping(patience=ES_PATIENCE, mode='min', verbose=1) 

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=model_file, save_best_only=True, mode='auto') 

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=RLR_PATIENCE, verbose=1,  mode='min')

    

    # 学習実行

    model.fit(

        [_x_train_image, _x_train], 

        _y_train, 

        validation_data=(

            [_x_valid_image, _x_valid], 

            _y_valid

        ),

        epochs = EPOCHS, 

        batch_size = BATCH_SIZE, 

        callbacks = [es, checkpoint, reduce_lr_loss]

    )



    # 精度が最も良いモデルで予測

    model.load_weights(model_file)

    valid_pred = model.predict([_x_valid_image, _x_valid], batch_size=PREDICT_BATCH).reshape((-1,1))

    mape_score = mean_absolute_percentage_error(_y_valid, valid_pred)

    scores.append(['socre_' + str(i), mape_score])



print(scores)

#sys.exit()
pred_names = []



# 最も精度の良かったモデルをトレーニングデータ全体で再学習し平均をとる

for i in range (0, LOOP_NUM):

    

    inputShape_vgg = train_images.shape[1:]

    inputShape_mlp = (len(x_train.columns.tolist()), )

    model = create_model(inputShape_vgg, inputShape_mlp)

    

    # 保存されたモデルを読み込む

    model.load_weights('best_model_' + str(i) + '.hdf5')

    model.trainable = False  #ホントはちゃんと再学習するレイヤを取捨選択するらしい

    model.compile(loss='mape', metrics=['mape'], optimizer=Adam(learning_rate=0.0001))

    # model.summary()

    

    # 再学習

    es = EarlyStopping(patience=ES_PATIENCE, mode='min', verbose=1) 

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=RLR_FACTOR, patience=RLR_PATIENCE, verbose=0, mode='min')

    model.fit(

        [train_images, x_train], 

        y_train, 

        epochs = EPOCHS, 

        verbose = 0,

        batch_size = BATCH_SIZE, 

        shuffle = True,

        callbacks = [es, reduce_lr_loss]

    )

    

    # 予測結果を取得

    pred = model.predict([pred_images, df_pred_copy.values], batch_size=PREDICT_BATCH).reshape((-1,1))

    pred_name = 'pred_' + str(i)

    pred_names.append(pred_name)

    df_pred[pred_name] = pred



# 各結果の平均をとって下２桁を四捨五入してからCSV出力

df_submit = df_pred.copy()

print(df_submit[pred_names].astype(int))



df_submit['price'] = df_submit[pred_names].mean(axis=1)

df_submit['price'] = df_submit['price'].round(-2).astype(int)

df_submit = df_submit[['id', 'price']]



df_submit.to_csv('MySubmission.csv', index=False)

print(df_submit)