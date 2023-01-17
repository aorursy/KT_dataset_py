import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras

img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw, train_size, val_size):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data, train_size=50000, val_size=5000)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

model1 = Sequential()
model1.add(Conv2D(12,(3,3),strides=2,activation='relu',input_shape=(img_rows,img_cols,1)))
model1.add(Conv2D(12,(3,3),strides=2,activation='relu'))
model1.add(Flatten())
model1.add(Dense(128,activation='relu'))
#model1.add(Dense(256,activation='relu'))
model1.add(Dense(num_classes, activation='softmax'))


model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model1.fit(x,y,batch_size=128, epochs=2, validation_split= 0.25)


model2 = Sequential()
model2.add(Conv2D(24,(3,3),activation='relu',input_shape=(img_rows,img_cols,1)))
model2.add(Conv2D(24,(3,3),activation='relu'))
model2.add(Conv2D(24,(3,3),activation='relu'))
model2.add(Flatten())
model2.add(Dense(128,activation='relu'))
#model2.add(Dense(256,activation='relu'))
#model2.add(Dense(128, activation='relu'))
model2.add(Dense(num_classes, activation='softmax'))


model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model2.fit(x,y,batch_size=128, epochs=4, validation_split= 0.25)
model3 = Sequential()
model3.add(Conv2D(24,(3,3),activation='relu',input_shape=(img_rows,img_cols,1)))
Dropout(0.5) #model3.add(Dropout(0.5))
model3.add(Conv2D(24,(3,3),activation='relu'))
Dropout(0.5) #model3.add(Dropout(0.5))
model3.add(Conv2D(24,(3,3), activation='relu'))
Dropout(0.5) #model3.add(Dropout(0.5))
model3.add(Flatten())
model3.add(Dense(128,activation='relu'))
#model1.add(Dense(256,activation='relu'))
model3.add(Dense(num_classes, activation='relu'))

model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model3.fit(x,y,batch_size=128, epochs=4, validation_split= 0.25)
