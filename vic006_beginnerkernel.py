import matplotlib.pyplot as plt
import numpy as np
import cv2
print(cv2.__version__)
import os
print(os.listdir("../input"))
X_train = np.load('../input/trainbeg.npy')


print(X_train)
X_test = np.load('../input/testbeg.npy')


print(X_test)
data_classes = ["antelope","bat","beaver","bobcat","buffalo","chihuahua","chimpanzee","collie","dalmatian","german+shepherd","grizzly+bear","hippopotamus","horse","killer+whale","mole","moose","mouse","otter","ox","persian+cat","raccoon","rat","rhinoceros","seal","siamese+cat","spider+monkey","squirrel","walrus","weasel","wolf"]

import pandas as pd
# tr_label = pd.read_csv('../input/train.csv')
# tr_label.head(10)


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

def conv_classifier(a):    
    model_input = Input(shape=(a, a,3))
    
    # Define a model architecture
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(model_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)       
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
import keras
model = conv_classifier(X_train.shape[1])

model.summary()
print(X_train.shape)


history = model.fit(x = X_train/255.,y = keras.utils.to_categorical(Y_train, num_classes=30),batch_size=128,epochs=30,validation_split=0.15, verbose=1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

print(model.evaluate(x=X_train/255., y=keras.utils.to_categorical(Y_train, num_classes=30), verbose=1))


y_train_predict = np.argmax(model.predict(x=X_train/255.),axis = 1)

print('\n',y_train_predict)

np.squeeze(Y_train)

print(Y_train)

print("Train accuracy : {}%".format(np.sum(y_train_predict == Y_train/(13000))))
tr_label = pd.read_csv('../input/sample_submission.csv')
tr_label.head(10)
y_test_predict = model.predict(x=X_test/255.)

print('\n',y_test_predict)
print(y_test_predict[1])

print(X_test/255.)
label_df = pd.DataFrame(data=y_test_predict, columns= data_classes)
label_df.head(10)
subm = pd.DataFrame()


te_label = pd.read_csv('../input/test.csv')


print(te_label['Image_id'])

subm['image_id'] = te_label['Image_id']

print(subm.head(10))
subm = pd.concat([subm, label_df], axis=1)

subm.to_csv('submitDL.csv',index = False)
subm
