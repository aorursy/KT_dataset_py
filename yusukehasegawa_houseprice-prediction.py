# 開始時間

from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=+9), 'JST')

print(datetime.now(JST))
import pandas as pd

import numpy as np

import datetime

import random

import glob

import cv2

import os

import sys



from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm import tqdm_notebook as tqdm



import tensorflow as tf

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, concatenate

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

import matplotlib.pyplot as plt

%matplotlib inline



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



# 乱数シード固定

seed_everything(2020)
!pip install kaggle
inputPath = '/kaggle/input/aiacademydeeplearning/'



# 画像サイズ

# サイズが小さすぎると細かい特徴量を把握できず、大きすぎると膨大な計算がなります。

# トレードオフを考えて選択しますが、普段は64～256を勧めます。

size = 128



# batch_size

# 	学習データのサンプル数（データ数）を指定する

# 	GPUのメモリサイズはそんなに大きくない

# 		家庭用で8GB、11GB。商用で16GB。

# 	メモリに載るサイズになるようにbatch_sizeを設定する

# 	最初は32をおススメ、次は64か16

batch_size = 32



# epoch

# 	多すぎると過学習しやすい

# 	最初は30をおススメ

epoch = 30



# kernel_initializer

# 	he_normal はKaggleコンペティションでも一般的に使われている

kernel_initializer = "he_normal"



# activation

# 	隠れ層はReLUをおススメ。良く使われる。

# 	他には Leaky ReLU か ELU で精度があがったコンペもある。

# 	回帰問題なら出力層のactivationはlinearを指定する

activation = "relu"



# Dropout

# 	一定割合のノードを不活性化（Drop）する、その割合を指定する。

# 	過学習を防ぐ（緩和）する、精度もあがる

# 	最初は0.2, 0.1-0.5を使う。0.5以上はほとんど使わない。

dropout_pct = 0.1



# Optimizer

# 	Adam

# 		テーブル・画像でも使われている

# 		安定的なOptimizerなので最初はこれ

# 		ハイパーパラメータチューニングせずとも良い精度が出る

# 	Radam

# 		最近注目されている

optimizer = "adam"



# Learning Rate

# 	0.001より大きくするのは経験的に良くない

# 	それより小さくすると改善する可能性ある

# 	小さすぎると学習が遅い

learning_rate = 0.001



# patience

# 	Early Stopping や ReduceLROnPlateau で何エポック改善しなかったら止めるか？の設定

patience = 2



# kernel_size

# 	どう決めればいいの？

# 		ふつうは 3 とか 5 が多い

# 		画像サイズが大きい時は大きくすることもある

kernel_size = (2, 2)



# filters

# 	出力畳み込みウィンドウkernelの数

# 	

# 	画像はfilterを徐々に増やしていく

# 	だいたい32→64→128、2倍ずつ増やしていく

start_filters = 32



# 読み込み

train_X = pd.read_csv(inputPath+'train.csv')

train_Y = train_X.price

train_X = train_X.drop(['price'], axis=1)

test_X = pd.read_csv(inputPath+'test.csv')



# 表示

display(train_X.head())



# 前処理（カテゴリ列 : zipcode）

#from category_encoders import OrdinalEncoder

#oe = OrdinalEncoder(cols=['zipcode'])

#train_X = oe.fit_transform(train_X)

#test_X = oe.transform(test_X)



# 正規化

from sklearn.preprocessing import StandardScaler

cols = ['bedrooms', 'bathrooms', 'area', 'zipcode']

scaler = StandardScaler()

for c in cols:

    train_X[c] = scaler.fit_transform(train_X.iloc[:, train_X.columns == c])

    test_X[c] = scaler.transform(test_X.iloc[:, test_X.columns == c])



# 表示

display(train_X.head())

%%time



def load_images(df,inputPath,size):

    images = []

    for i in df['id']:

        print(str(i)+", ", end="")

        

        basePath = os.path.sep.join([inputPath, "{}_*".format(i)])

        housePaths = sorted(list(glob.glob(basePath)))

        

        input_images = []       

        output_image = np.zeros((size*2, size*2, 3), dtype="uint8")

        

        # bathroom, bedroom, frontal, kitchen の 4つ画像を読み込んでリサイズ

        for housePath in housePaths:

            image = cv2.imread(housePath)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (size, size))

            

#            plt.figure(figsize=(8,4))

#            plt.imshow(image)

           

            input_images.append(image)

            

        # bathroom, bedroom, frontal, kitchen の 4つ画像をタイル状に1枚の画像にする

        output_image[0:size,      0:size]      = input_images[0]

        output_image[0:size,      size:size*2] = input_images[1]

        output_image[size:size*2, size:size*2] = input_images[2]

        output_image[size:size*2, 0:size]      = input_images[3]



#        plt.figure(figsize=(8,4))

#        plt.imshow(output_image)

        

        images.append(output_image)

        

    # 最小値0、最大値1に変換する正規化のため、255で割る

    return np.array(images) / 255.0



# load images

print("\nload train images...")

train_images = load_images(train_X,inputPath+'train_images/',size)

print("\nload test images...")

test_images = load_images(test_X,inputPath+'test_images/',size)

def create_multimodal_nn(cnn_inputShape, mlp_inputDim):

    # create CNN model

    cnn_model = Sequential()



    cnn_model.add(Conv2D(filters=start_filters, kernel_size=kernel_size, strides=(1, 1), padding='valid',

                     activation=activation, kernel_initializer=kernel_initializer, input_shape=cnn_inputShape))

    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(BatchNormalization())

    cnn_model.add(Dropout(dropout_pct))



    cnn_model.add(Conv2D(filters=start_filters*2, kernel_size=kernel_size, strides=(1, 1), padding='valid', 

                     activation=activation, kernel_initializer=kernel_initializer))

    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(BatchNormalization())

    cnn_model.add(Dropout(dropout_pct))

    

    cnn_model.add(Conv2D(filters=start_filters*4, kernel_size=kernel_size, strides=(1, 1), padding='valid', 

                     activation=activation, kernel_initializer=kernel_initializer))

    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(BatchNormalization())

    cnn_model.add(Dropout(dropout_pct))



    cnn_model.add(Conv2D(filters=start_filters*8, kernel_size=kernel_size, strides=(1, 1), padding='valid', 

                     activation=activation, kernel_initializer=kernel_initializer))

    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(BatchNormalization())

    cnn_model.add(Dropout(dropout_pct))



    cnn_model.add(Flatten())

    

    # create MLP model

    mlp_model = Sequential()

    mlp_model.add(Dense(units=2046, input_dim=mlp_inputDim, 

                    kernel_initializer=kernel_initializer,activation=activation))    

    mlp_model.add(Dropout(dropout_pct))

    mlp_model.add(Dense(units=1024,  kernel_initializer=kernel_initializer,activation=activation))

    mlp_model.add(Dropout(dropout_pct))

    mlp_model.add(Dense(units=512,  kernel_initializer=kernel_initializer,activation=activation))

    mlp_model.add(Dropout(dropout_pct))

    mlp_model.add(Dense(units=256, kernel_initializer=kernel_initializer, activation=activation))     

    mlp_model.add(Dropout(dropout_pct))

    mlp_model.add(Dense(units=32, kernel_initializer=kernel_initializer, activation=activation))     

    mlp_model.add(Dropout(dropout_pct))



    # merge

    merge_input = concatenate([cnn_model.output, mlp_model.output])

    

    x = Dense(units=256, activation=activation,kernel_initializer=kernel_initializer)(merge_input)

    x = Dense(units=32, activation=activation,kernel_initializer=kernel_initializer)(x)

    x = Dense(units=1, activation='linear')(x)

    

    model = Model(inputs=[cnn_model.input, mlp_model.input], outputs=x)

    

    model.compile(loss='mape', optimizer=optimizer, metrics=['mape']) 

    

    return model
# 'id' 列は教師データから除外

train_X = train_X.drop(['id'], axis=1)

test_X = test_X.drop(['id'], axis=1)
%%time



def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100





#CVしてスコアを見る

scores = []



kf = KFold(n_splits=5, shuffle=True)



for i, (train_ix, valid_ix) in enumerate(kf.split(train_X)):

    print("============ starting CV#" + str(i) + "==========================================================")

    train_X_, train_Y_ = train_X.iloc[train_ix].values, train_Y.iloc[train_ix].values

    valid_X_, valid_Y_ = train_X.iloc[valid_ix].values, train_Y.iloc[valid_ix].values

    

    train_image_X_ = train_images[train_ix]

    valid_image_X_ = train_images[valid_ix]

    

    # callback parameter

    filepath = "cnn_best_model_CV"+str(i)+".hdf5" 

    es = EarlyStopping(patience=patience, mode='min', verbose=1) 

    checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=patience, verbose=1,  mode='min')



    # 訓練実行

    cnn_inputShape = (size*2, size*2, 3)

    mlp_inputDim = train_X_.shape[1]

    model = create_multimodal_nn(cnn_inputShape, mlp_inputDim)

    

    model.fit(

        [train_image_X_, train_X_], train_Y_,

        validation_data=([valid_image_X_, valid_X_], valid_Y_),

        epochs=epoch, 

        batch_size=batch_size,

        callbacks=[es, checkpoint, reduce_lr_loss]

    )



    # load best model weights

    if os.path.exists(filepath):

        model.load_weights(filepath)



    # 評価

    valid_pred = model.predict([valid_image_X_, valid_X_], batch_size=batch_size).reshape((-1,1))

    mape_score = mean_absolute_percentage_error(valid_Y_, valid_pred)

    scores.append(mape_score)

    

print(scores)

print(sum(scores)/len(scores)) 
%%time

import itertools



predictions = pd.DataFrame()



for i in range(3):

    print("============ starting seed#" + str(i) + "==========================================================")

    seed_everything(i)

    

    # callback parameter

    filepath = "cnn_best_model_Seed"+str(i)+".hdf5" 

    es = EarlyStopping(patience=patience, mode='min', verbose=1) 

    checkpoint = ModelCheckpoint(monitor='loss', filepath=filepath, save_best_only=True, mode='auto') 

    reduce_lr_loss = ReduceLROnPlateau(monitor='loss',  patience=patience, verbose=1,  mode='min')



    # 訓練実行

    cnn_inputShape = (size*2, size*2, 3)

    mlp_inputDim = train_X_.shape[1]

    model = create_multimodal_nn(cnn_inputShape, mlp_inputDim)

    model.fit(

        [train_images, train_X], train_Y,

        epochs=epoch, 

        batch_size=batch_size,

        callbacks=[es, checkpoint, reduce_lr_loss]

    )



    # load best model weights

    if os.path.exists(filepath):

        model.load_weights(filepath)



    # 予測

    predictions['Seed#'+str(i)] = list(itertools.chain.from_iterable(model.predict([test_images, test_X], batch_size=batch_size).reshape((-1,1))))



predictions
df_submission = pd.read_csv(inputPath+'sample_submission.csv')

df_submission.price = predictions.mean(axis=1).values

df_submission.to_csv('submission.csv', index=False)

df_submission
message = 'CNN Layers:4, MLP Layers:5, ' + 'size:' + str(size) + ', batch_size:' + str(batch_size) + ', epoch:' + str(epoch) + ', kernel_initializer:' + str(kernel_initializer) + ', activation:' + str(activation) + ', dropout_pct:' + str(dropout_pct) + ', optimizer:' + str(optimizer) + ', learning_rate:' + str(learning_rate) + ', patience:' + str(patience) + ', kernel_size:' + str(kernel_size) + ', start_filters:' + str(start_filters)

print(message)

!export KAGGLE_USERNAME=xxxxx; export KAGGLE_KEY=xxxxx; kaggle competitions submit -c aiacademydeeplearning -f submission.csv -m "$message"
model.summary()
plot_model(model, to_file='nn_model.png')
# 終了時間

from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=+9), 'JST')

print(datetime.now(JST))