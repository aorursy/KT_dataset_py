import numpy as np
import pandas as pd
from matplotlib import pylab as plt
import cv2

import tensorflow as tf
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, Dense, MaxPool2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import optimizers
df_total = pd.read_csv('../input/facial-keypoints-detection/training/training.csv')
df_total.shape
df_total.head(1)
df_total.isna().sum()
X_total_str_s = df_total.iloc[:, -1]
X_total_str_s.shape
def parse_img_str(img_str):
    img_values = img_str.split(' ')
    img_values = [int(i) for i in img_values]
    img_width = int(np.sqrt(len(img_values)))
    img = np.array(img_values).reshape(img_width, img_width)
    
    return img
X_total_s = X_total_str_s.apply(parse_img_str)
X_total = np.stack(X_total_s.values)
X_total.shape
Y_total = df_total.iloc[:, :-1].values
Y_total.shape
Y_total[4000]
def draw_face_keypoints(img, label):
    img = img.astype(np.uint8)

    if label is not None:
        for i in range(0, 30, 2):
            x = int(label[i])
            y = int(label[i+1])
            cv2.circle(img, (x, y), 0, (255, 0, 0), 1)

    plt.rcParams['image.cmap'] = 'gray'
    plt.imshow(img)
example_img = X_total[8]
example_label = Y_total[8]
draw_face_keypoints(example_img, example_label)
def flip_dataset(X, Y):
    X_flip = X[:, :, ::-1]
    
    Y_flip = Y.copy()
    for i in range(0, 30, 2):
        Y_flip[:, i] = 96 - Y_flip[:, i]
        
    return X_flip, Y_flip
def data_augmentation(X, Y):
    X_flip, Y_flip = flip_dataset(X, Y)
    X_total = np.concatenate([X, X_flip], 0)
    Y_total = np.concatenate([Y, Y_flip], 0)
    
    random_indexs = np.random.permutation(X_total.shape[0])
    X_total = X_total[random_indexs]
    Y_total = Y_total[random_indexs]
    
    return X_total, Y_total
df_test = pd.read_csv('../input/facial-keypoints-detection/test (8)/test.csv')
df_test.head(1)
X_test_str_s = df_test['Image']
X_test_s = X_test_str_s.apply(parse_img_str)
X_test = X_test_s.values
X_test = np.stack(X_test)
X_test.shape
example_test = X_test[10]
draw_face_keypoints(example_test, None)
random_indexs = np.random.permutation(X_total.shape[0])
X_total_shuffle = X_total[random_indexs]
Y_total_shuffle = Y_total[random_indexs]
dev_size = int(X_total.shape[0] * 0.3)
X_dev = X_total_shuffle[:dev_size].reshape((-1, 96, 96, 1))
Y_dev = Y_total_shuffle[:dev_size]
X_dev.shape, Y_dev.shape
X_train = X_total_shuffle[dev_size:]
Y_train = Y_total_shuffle[dev_size:]
X_train.shape, Y_train.shape
X_total_aug, Y_total_aug = data_augmentation(X_total, Y_total)
X_total_aug = X_total_aug.reshape((-1, 96, 96, 1))
X_total_aug.shape, Y_total_aug.shape
X_train_aug, Y_train_aug = data_augmentation(X_train, Y_train)
X_train_aug = X_train_aug.reshape((-1, 96, 96, 1))
X_train_aug.shape, Y_train_aug.shape
mean_img = np.mean(X_train_aug, axis=0)
mean_img.shape
X_train_norm = X_train_aug - mean_img
Y_train_norm = Y_train_aug
X_dev_norm = X_dev - mean_img
Y_dev_norm = Y_dev
def model():
    X_input = Input(shape=(96, 96, 1), name='input')
    
    X = Conv2D(32, (3, 3), padding='same', use_bias=False, name='conv_1')(X_input)
    X = BatchNormalization(name='bn_1')(X)
    X = Activation('relu', name='relu_1')(X)
    X = MaxPool2D((2, 2), name='maxpool_1')(X)
    
    X = Conv2D(64, (3, 3), padding='same', use_bias=False, name='conv_2')(X)
    X = BatchNormalization(name='bn_2')(X)
    X = Activation('relu', name='relu_2')(X)
    X = MaxPool2D((2, 2), name='maxpool_2')(X)

    
    X = Conv2D(128, (3, 3), padding='same', use_bias=False, name='conv_3')(X)
    X = BatchNormalization(name='bn_3')(X)
    X = Activation('relu', name='relu_3')(X)
    X = MaxPool2D((2, 2), name='maxpool_3')(X)
    
    X = Conv2D(256, (3, 3), padding='same', use_bias=False, name='conv_4')(X)
    X = BatchNormalization(name='bn_4')(X)
    X = Activation('relu', name='relu_4')(X)
    X = MaxPool2D((2, 2), name='maxpool_4')(X)

    X = Flatten()(X)
    
    X = Dense(30, activation='relu', name='output')(X)
    
    model = Model(inputs=X_input, outputs=X, name='face_keypoints_net')
    
    return model
def multi_loss(y_true, y_pred):
    bool_mask = y_true > 0
    y_true = tf.boolean_mask(y_true, bool_mask)
    y_pred = tf.boolean_mask(y_pred, bool_mask)
    
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
def multi_rmse(y_true, y_pred):
    return tf.sqrt(multi_loss(y_true, y_pred))
facenet = model()
adam = optimizers.Adam(lr=0.01)
facenet.compile(adam, loss=[multi_loss], metrics=[multi_rmse])
facenet.fit(x=X_train_norm, y=Y_train_norm, batch_size=128, epochs=50)
facenet.evaluate(x=X_train_norm, y=Y_train_norm, batch_size=128)
facenet.evaluate(x=X_dev_norm, y=Y_dev_norm, batch_size=128)
df_test = pd.read_csv('../input/facial-keypoints-detection/test (8)/test.csv')
df_test.head(1)
test_img_strs = df_test['Image']
test_img_strs.shape
test_imgs_s = test_img_strs.apply(parse_img_str)
test_imgs = np.stack(test_imgs_s)
test_imgs = test_imgs.reshape((-1, 96, 96, 1))
X_test_norm = test_imgs - mean_img
predicts = facenet.predict(X_test_norm)
predicts.shape
draw_face_keypoints(test_imgs[7].reshape((96, 96)), predicts[7])
keys = df_total.columns[:-1]
predicts_dicts = [dict(zip(keys, keypoints)) for keypoints in predicts]
df_sample = pd.read_csv('../input/facial-keypoints-detection/SampleSubmission.csv')
df_sample.head(3)
df_look_table = pd.read_csv('../input/facial-keypoints-detection/IdLookupTable.csv')
df_look_table.head(3)
for i in range(len(df_look_table)):
    imageId = df_look_table.iloc[i]['ImageId']
    featureName = df_look_table.iloc[i]['FeatureName']
    
    location = predicts_dicts[imageId - 1][featureName]
    
    df_sample.iloc[i, 1] = location
    
df_sample.head(10)
df_sample.to_csv('predict_result_0.csv')