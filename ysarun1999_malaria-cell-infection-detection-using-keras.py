import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from keras.models import Sequential

from keras.utils import to_categorical

from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
data=[]

labels=[]

uninfected_path = os.listdir('../input/cell_images/cell_images/Uninfected/')

for i in uninfected_path:

    try:

        image = cv2.imread('../input/cell_images/cell_images/Uninfected/' + i)

        image_from_array = Image.fromarray(image,'RGB')

        sized_image = image_from_array.resize((64,64))

        data.append(np.array(sized_image))

        labels.append(0)

    except Exception as e:

        print(e)
parasitised_path = os.listdir('../input/cell_images/cell_images/Parasitized/')

for i in parasitised_path:

    try:

        image = cv2.imread('../input/cell_images/cell_images/Parasitized/' + i)

        image_from_array = Image.fromarray(image,'RGB')

        sized_image = image_from_array.resize((64,64))

        data.append(np.array(sized_image))

        labels.append(1)

    except Exception as e:

        print(e)
X = np.array(data)

y = np.array(labels)
print(X.shape,y.shape)
X = X.astype('float32')

X/=255
y = y.astype('int32')

y = to_categorical(y,num_classes=2)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)
model = Sequential()

model.add(Conv2D(32,(3,3),padding = 'Same',activation = 'relu',input_shape = (64,64,3)))

model.add(Conv2D(32,(3,3),padding = 'Same',activation = 'relu'))

model.add(MaxPooling2D(pool_size = (3,3)))

model.add(Dropout(0.25))



model.add(Conv2D(64,(3,3),padding = 'Same',activation = 'relu'))

model.add(Conv2D(64,(3,3),activation = 'relu'))

model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512,activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(2,activation = 'softmax'))
model.compile(optimizer = 'RMSprop',loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.summary()
history = model.fit(X_train,y_train,batch_size=86,epochs=30,verbose=1,validation_data=(X_test,y_test))

model.evaluate(X_test,y_test)
plt.figure(figsize = (20,8))

plt.plot(range(30), history.history['acc'], label = 'Training Accuracy')

plt.plot(range(30), history.history['loss'], label = 'Taining Loss')

plt.xlabel("Epochs")

plt.ylabel('Accuracy/Loss Value')

plt.legend(loc = "best")