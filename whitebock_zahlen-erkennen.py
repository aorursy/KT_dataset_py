import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

train = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

train_labels = train["label"]
train_data = train.drop(labels = ["label"],axis = 1) 
del train

# Graustufen Effekt um die Beleuchtung anzugleichen 
train_data = train_data / 255.0
test_data = test_data / 255.0

# Alle Bilder gleich groß skalieren
train_data = train_data.values.reshape(-1,28,28,1)
test_data = test_data.values.reshape(-1,28,28,1)

# Label von Zahlen zu Vektoren konvertieren
train_labels = to_categorical(train_labels, num_classes = 10)

# Einen teil der Daten nehmen wir nicht zum lernen, sondern zum überprüfen
train_data, validation_data, train_labels, validation_labels = train_test_split(train_data, train_labels, test_size = 0.1, random_state=37)

# Hier ein Beispiel der aktuellen Daten
img = plt.imshow(train_data[0][:,:,0])
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=10,zoom_range = 0.1,width_shift_range=0.1,height_shift_range=0.1)
datagen.fit(train_data)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3,verbose=1,factor=0.5,min_lr=0.00001)

history = model.fit_generator(datagen.flow(train_data,train_labels, batch_size=2), epochs=3, validation_data = (validation_data,validation_labels),
                              verbose=2, steps_per_epoch=train_data.shape[0],callbacks=[learning_rate_reduction])
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
import numpy as np

results = model.predict(test_data)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name='Label')

submission = pd.concat([pd.Series(range(1,28001),name = 'ImageId'),results],axis = 1)
submission.to_csv('submission.csv',index=False)