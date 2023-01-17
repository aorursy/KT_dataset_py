import matplotlib.pyplot as plt
import numpy as np
import cv2
print(cv2.__version__)
import os
print(os.listdir("../input"))
# X_train = np.load('../input/trainbeg.npy')

inp_dm = 150
# print(X_train)

X_train = np.zeros((13000,inp_dm,inp_dm,3))

for i in range(13000):
    image = cv2.imread('../input/train/train/Img-{}.jpg'.format(i+1))
    #print(image.shape)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     plt.imshow(gray_image)
#     plt.show()
#     plt.imshow(image)
#     plt.show()
    resized_image = cv2.resize(image, (inp_dm, inp_dm)) 
    X_train[i] =  resized_image
    if i % 100 == 0:
        print(i)

print(X_train)
# X_test = np.load('../input/testbeg.npy')


# print(X_test)

X_test = np.zeros((6000,inp_dm,inp_dm,3))

for i in range(6000):
    image = cv2.imread('../input/test/test/Img-{}.jpg'.format(i+1))
    #print(image.shape)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     plt.imshow(gray_image)
#     plt.show()
#     plt.imshow(image)
#     plt.show()
    resized_image = cv2.resize(image, (inp_dm, inp_dm)) 
    X_test[i] =  resized_image
    if i % 100 == 0:
        print(i)

print(X_test)
data_classes = ["antelope","bat","beaver","bobcat","buffalo","chihuahua","chimpanzee","collie","dalmatian","german+shepherd","grizzly+bear","hippopotamus","horse","killer+whale","mole","moose","mouse","otter","ox","persian+cat","raccoon","rat","rhinoceros","seal","siamese+cat","spider+monkey","squirrel","walrus","weasel","wolf"]

import pandas as pd
# tr_label = pd.read_csv('../input/train.csv')
# tr_label.head(10)

print(X_train)
Y_train =  np.load('../input/trainLabels.npy')

print(Y_train.shape)

Y_train = Y_train.reshape(Y_train.shape[0])


np.squeeze(Y_train)
print(Y_train.shape)
print(Y_train)

# te_label = pd.read_csv('/media/vedavikas/New Volume1/DL/meta-data/test.csv')
# te_label.head(10)
from sklearn.model_selection import cross_val_score
from keras.models import Sequential, load_model, Model
from keras.layers import Input, BatchNormalization
from keras.layers import Dense, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D,Dropout
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
import keras
def conv_classifier(a):    
    model_input = Input(shape=(a, a,3))
    
    # Define a model architecture
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(model_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(model_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)       
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)       
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)    
    x = Dropout(0.25)(x)

    
    y1 = Dense(30, activation='softmax')(x)
    
    model = Model(inputs=model_input, outputs= y1)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# model = conv_classifier(X_train.shape[1])

# model.summary()
# print(X_train.shape)

# callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3)]


# history = model.fit(x = X_train/255.,y = keras.utils.to_categorical(Y_train, num_classes=30),batch_size=128,epochs=50,validation_split=0.2,callbacks=callbacks_list,verbose=1)
# plt.plot(history.history['acc'])
# #plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')

# print(model.evaluate(x=X_train/255., y=keras.utils.to_categorical(Y_train, num_classes=30), verbose=1))


# y_train_predict = np.argmax(model.predict(x=X_train/255.),axis = 1)
# from sklearn.metrics import accuracy_score

# print('\n',y_train_predict)

# np.squeeze(Y_train)

# print(Y_train)

# # print("Train accuracy : {}%".format(np.sum(y_train_predict == Y_train/(13000))))

# print("Train accuracy : {}%".format(accuracy_score(Y_train,y_train_predict)))
# tr_label = pd.read_csv('../input/sample_submission.csv')
# tr_label.head(10)
# y_test_predict = model.predict(x=X_test/255.)

# print('\n',y_test_predict)
# label_df = pd.DataFrame(data=y_test_predict, columns= data_classes)
# label_df.head(10)

# subm = pd.DataFrame()


# te_label = pd.read_csv('../input/test.csv')


# print(te_label['Image_id'])

# subm['image_id'] = te_label['Image_id']

# print(subm.head(10))
# subm = pd.concat([subm, label_df], axis=1)
# subm.to_csv('submitcnn.csv',index = False)

# subm

vgg_model_path = '../input/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
from keras.applications import VGG16
conv_base = VGG16(weights=vgg_model_path,include_top=False,input_shape=(inp_dm,inp_dm, 3))
conv_base.trainable = False

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        layer.trainable = True
    if layer.name == 'block4_conv1':
        layer.trainable = True    
    else:
        layer.trainable = False

conv_base.summary()
def VGG16_classifier():    
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten()) 
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='softmax'))
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
model1 = VGG16_classifier()
callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3)]

history_vgg = model1.fit(x = X_train/255.,y = keras.utils.to_categorical(Y_train, num_classes=30),batch_size=64,epochs=25,validation_split=0.1,callbacks = callbacks_list, verbose=1)

from sklearn.metrics import accuracy_score
y_train_predict = np.argmax(model1.predict(x=X_train/255.),axis = 1)

print('\n',y_train_predict)

np.squeeze(Y_train)

print(Y_train)

print("Train accuracy : {}%".format(accuracy_score(Y_train,y_train_predict)))
y_test_predict = model1.predict(x=X_test/255.)

print('\n',y_test_predict)

label_df = pd.DataFrame(data=y_test_predict, columns= data_classes)

label_df.head(10)

subm = pd.DataFrame()


te_label = pd.read_csv('../input/test.csv')


print(te_label['Image_id'])

subm['image_id'] = te_label['Image_id']

print(subm.head(10))
subm = pd.concat([subm, label_df], axis=1)

subm.to_csv('submitvgg.csv',index = False)

print(subm)
