import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow import keras  # tensorflow is our library, keras on top of it makes it simpler
from keras.optimizers import Adam
from keras import regularizers
from keras.activations import relu
train_data = pd.read_csv('../input/train.csv')
train_data.shape
train_data.head()   # looking to our DataFrame
train_data.info()    # we see that they are all integers
train_data.describe()   # we see that it needs to be scaled (now it's between 0 and 255 )
train_y = train_data["label"] 

train_data.drop(["label"],axis=1,inplace=True)

train_x = train_data
train_x = train_x.values.reshape(-1,28,28,1)
train_y = train_y.values
from keras.utils.np_utils import to_categorical
train_y = to_categorical(train_y)
train_y.shape
len(train_y[5])
train_x.shape
train_x = train_x / 255.0
train_x.shape
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,AveragePooling2D,Dense,Dropout,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from tensorflow.nn import leaky_relu 
model = Sequential()

model.add(Conv2D(32, kernel_size = (5,5),padding = 'same',activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(32,kernel_size=(5,5),padding="same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),padding="same"))
model.add(Dropout(0.25))

model.add(Conv2D(32,kernel_size=(3,3),padding="same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),padding="same"))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(32,activation=leaky_relu))
model.add(Dropout(0.25))
model.add(Dense(20,activation=leaky_relu))

model.add(Dense(10, activation = "softmax"))
model.summary()
model.compile(Adam(lr=0.0003),loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
datagen = ImageDataGenerator(
       # rotation_range=0.5, 
       # zoom_range = 0.5, 
       # width_shift_range=0.5,  
       # height_shift_range=0.5
)

datagen.fit(train_x)
model.fit_generator(datagen.flow(train_x,train_y, batch_size=80),steps_per_epoch=525,epochs=30)
test_x = pd.read_csv('../input/test.csv')

test_x.head(10)
test_x = test_x.values.reshape(-1,28,28,1)
test_x = test_x / 255.0
predictions = model.predict(test_x)
predictions[354]
pred = np.argmax(predictions, axis=1)
import matplotlib.pyplot as plt
plt.imshow(test_x[358][:,:,0],cmap='gray')
plt.show()
pred[358]
my_submission = pd.DataFrame({'ImageId': range(1,len(test_x)+1) ,'Label':pred })

my_submission.to_csv("cnn_results3.csv",index=False)
my_submission.head(10)
#efe ergün