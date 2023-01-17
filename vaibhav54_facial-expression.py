import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import cv2



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
df = pd.read_csv('/kaggle/input/facial-expression/fer2013.csv')

df.head()
X_train, X_test, y_train, y_test = list(), list(), list(), list()



for i in df.index: 

    

    temp = df['pixels'][i]

    temp = np.fromstring(temp, dtype=int, sep=' ')

    temp = temp / 255.0

    temp = np.reshape(temp, (48,48,1))

    if(df['Usage'][i] == 'Training'):    

        X_train.append(temp)  

        y_train.append(df['emotion'][i])

    if(df['Usage'][i] == 'PrivateTest'):    

        X_test.append(temp)  

        y_test.append(df['emotion'][i])    

len(X_train), len(X_test)
img = X_train[10].reshape(48,48)

plt.imshow(img, cmap='gray')
X_train, X_test = np.array(X_train), np.array(X_test)
X_train.shape, X_train[34].shape
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0, 1))



# X_train = scaler.fit_transform(X_train)

# X_test = scaler.fit_transform(X_test)
from keras.utils import to_categorical



y_test = to_categorical(y_test)

y_train = to_categorical(y_train)
from keras.models import Sequential 

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation

from keras.callbacks import EarlyStopping

from keras.layers.normalization import BatchNormalization

model = Sequential()



model.add(Conv2D( 32, (3,3), padding='Same', input_shape=(48,48,1), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2,2)))





model.add(Conv2D( 64, (3,3), padding='Same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2,2)))

          

model.add(Conv2D( 128, (3,3), padding='Same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D( 256, (3,3), padding='Same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D( 512, (3,3), padding='Same', activation='relu'))





# model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D( 1024, (3,3), padding='Same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2,2)))





model.add(Dropout(0.5))



model.add(Flatten())



model.add(Dense(1024,activation='relu', kernel_regularizer='l2'))     

model.add(BatchNormalization())

model.add(Dense(512,activation='relu', kernel_regularizer='l2'))

model.add(Dense(256,activation='relu', kernel_regularizer='l2'))



model.add(Dense(32))          

 

          

# model.add(Activation('relu'))

# model.add(Dropout(0.5))



# model.add(Dense(1, activation='sigmoid'))

model.add(Dense(7, activation='softmax'))  

          

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])



model.summary()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),

                    epochs=25, verbose=1,batch_size=128) 



# history = model.fit(X_train,y_train,validation_data=(X_val,y_v al), batch_size=128, epochs=30, verbose=1)
weights = model.get_weights()

for i in weights:

    print(np.max(i))
plt.figure(figsize=(10,6))

plt.plot(history.history['val_loss'], color='red', label='test')

plt.plot(history.history['loss'], color='blue', label='train')

plt.plot(grid=True)

plt.legend()

plt.show()
plt.figure(figsize=(10,6))

plt.plot(history.history['val_accuracy'], color='red', label='test')

plt.plot(history.history['accuracy'], color='blue', label='train')

plt.legend()

plt.show()
pred = model.predict(X_test)

pred
i = 12



print(label_map[np.argmax(pred[i])])



img = X_test[78].reshape(48,48)

plt.imshow(img,cmap='gray')
plt.figure(figsize=(12,10))





for num, val in enumerate(X_test[10:16]):

#     encoded = encode(val)

#     pred_cap = greedy_search(encoded.reshape(1,4096))

#     pred = model.predict(i)

    cap = label_map[np.argmax(pred[num])]

    img = val.reshape(48,48)

    plt.subplot(5,2,num+1)

    plt.title(cap)

    plt.axis('off')

    plt.imshow(img,cmap='gray')