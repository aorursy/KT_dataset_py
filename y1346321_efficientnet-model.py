!pip install -U git+https://github.com/qubvel/efficientnet
import os

import numpy as np

import pandas as pd



import tensorflow as tf

import keras



from keras.preprocessing import image



from keras.layers import Conv2D,Dropout,Dense,Flatten

from keras.models import Sequential



from keras.layers.advanced_activations import LeakyReLU

from keras.models import Sequential, Model

from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D



import efficientnet.keras as efn



print(tf.__version__)
INPUT_PATH = '/kaggle/input'

IMAGE_DIR = os.path.join(INPUT_PATH, 'faces_images/faces_images')

TRAIN_CSV_PATH = os.path.join(INPUT_PATH, 'train_vision.csv')

TEST_CSV_PATH = os.path.join(INPUT_PATH, 'test_vision.csv')



os.listdir(INPUT_PATH)
train_csv = pd.read_csv(TRAIN_CSV_PATH)

test_csv = pd.read_csv(TEST_CSV_PATH)



train_csv.head()
train_csv['label'].hist()
def load_image_array(image_path):

    img = image.load_img(image_path, target_size=(128, 128))

    image_array = image.img_to_array(img)

    

    return image_array

    

def load_image_data(filenames):

    image_data = [load_image_array(os.path.join(IMAGE_DIR, filename)) for filename in filenames]

    

    return np.array(image_data).astype(float)/255.
X_train = load_image_data(train_csv['filename'].values)

y_train = pd.get_dummies(train_csv['label'].astype(str))
X_train.shape
y_train[:10]
for column in ['glasses', 'children', 'femail']:

    y_train[column] = 0
y_train.loc[(y_train['2']==1) | (y_train['5']==1), 'glasses'] = 1

y_train.loc[(y_train['3']==1) | (y_train['6']==1), 'children'] = 1

y_train.loc[(y_train['4']==1) | (y_train['5']==1) | (y_train['6']==1), 'femail'] = 1



y_train.head()
y_train_glasses = y_train['glasses']

y_train_age = y_train['children']

y_train_gender = y_train['femail']
def get_model(input_shape=(128,128,3)):

    model = Sequential()



    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=input_shape))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    

    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))



    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())



    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))



    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())



    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))



    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

    # model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())



    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))



    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())



    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))



    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    

    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha=0.1))

    model.add(BatchNormalization())

    

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.1))

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.1))

    model.add(Dense(1, activation='sigmoid'))

    

    model.compile(optimizer='adam',

                  loss='binary_crossentropy',

                  metrics=['accuracy'])

    

    return model

def get_model_efficient(input_shape=(128,128,3)):

    base_model = efn.EfficientNetB5(weights='imagenet', include_top=False, input_shape=input_shape)

    base_model.trainable = False

    

    x = base_model.output

    x = Flatten()(x)

    x = Dense(1024, activation="relu")(x)

    x = Dropout(0.5)(x)

    

    #y_pred = Dense(6, activation="softmax")(x)

    y_pred = Dense(1, activation="sigmoid")(x)

    

    #loss = 'categorical_crossentropy'

    loss = 'binary_crossentropy'

    

    model = Model(input=base_model.input, output=y_pred)

    

    for base_layer in model.layers[:-3]:

        base_layer.trainable = True

        

    model.compile(optimizer='adam',

                  loss=loss,

                  metrics=['accuracy'])

    

    

    return model

model_glasses = get_model_efficient()

model_age = get_model_efficient()

model_gender = get_model_efficient()
models = [model_glasses, model_age, model_gender]

y_trains = [y_train_glasses, y_train_age, y_train_gender]



for i in range(3):

    models[i].fit(X_train, y_trains[i], 

                  epochs=1, batch_size=32, validation_split=0.2)
model_gender.fit(X_train, y_train_gender, 

                 epochs=1, batch_size=32, validation_split=0.2)
X_test = load_image_data(test_csv['filename'].values)
predict_glasses = model_glasses.predict(X_test).reshape(-1)

predict_age = model_age.predict(X_test).reshape(-1)

predict_gender = model_gender.predict(X_test).reshape(-1)
y_pred = pd.DataFrame({'glass': list(predict_glasses), 

                       'children': list(predict_age), 

                       'femail': list(predict_gender)})

y_pred
y_pred['predict'] = 0



y_pred.loc[(y_pred['glass']<0.5)&(y_pred['children']<0.5)&(y_pred['femail']<0.5), 'predict'] = 1

y_pred.loc[(y_pred['glass']>=0.5)&(y_pred['children']<0.5)&(y_pred['femail']<0.5), 'predict'] = 2

y_pred.loc[(y_pred['glass']>=0.5)&(y_pred['children']>=0.5)&(y_pred['femail']<0.5), 'predict'] = 2

y_pred.loc[(y_pred['glass']<0.5)&(y_pred['children']>=0.5)&(y_pred['femail']<0.5), 'predict'] = 3



y_pred.loc[(y_pred['glass']<0.5)&(y_pred['children']<0.5)&(y_pred['femail']>=0.5), 'predict'] = 4

y_pred.loc[(y_pred['glass']>=0.5)&(y_pred['children']<0.5)&(y_pred['femail']>=0.5), 'predict'] = 5

y_pred.loc[(y_pred['glass']>=0.5)&(y_pred['children']>=0.5)&(y_pred['femail']>=0.5), 'predict'] = 5

y_pred.loc[(y_pred['glass']<0.5)&(y_pred['children']>=0.5)&(y_pred['femail']>=0.5), 'predict'] = 6



y_pred.head(50)
submit = pd.DataFrame({'prediction': y_pred['predict'].values})

submit
submit.to_csv('submission_19.csv', index=False)