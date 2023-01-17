from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

import os
path = '/kaggle/input/devnagri-hindi-dataset/DevanagariHandwrittenCharacterDataset'
os.listdir('/kaggle/input/devnagri-hindi-dataset/DevanagariHandwrittenCharacterDataset/')
train_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip = False, validation_split = 0.13)
test_datagen = ImageDataGenerator(rescale = 1./255)
train_set = train_datagen.flow_from_directory(path + '/Train', target_size = (64, 64), batch_size = 32, class_mode = 'categorical', subset='training')
validation_set = train_datagen.flow_from_directory(path + '/Train', target_size = (64, 64), batch_size = 32, class_mode = 'categorical', subset='validation')
test_set = test_datagen.flow_from_directory(path + '/Test', target_size = (64, 64), batch_size = 32, class_mode = 'categorical' )
Labels = train_set.class_indices

print(Labels)

num_classes = len(Labels)
Characters = 'क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह क्ष त्र ज्ञ ० १ २ ३ ४ ५ ६ ७ ८ ९'
Characters = Characters.split(' ')
key_list = list(Labels.keys())
import re

def atoi(text):

    return int(text) if text.isdigit() else text

def natural_keys(text):

    return [ atoi(c) for c in re.split('(\d+)',text) ]

ls = os.listdir(path+'/Train')

ls.sort(key = natural_keys)
print(ls,end  ='')
Cha_Uni = {}

for j,i in enumerate(ls):

    Cha_Uni[i] = Characters[j]
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(64, 64, 3), activation = 'relu'))

model.add(Conv2D(64, (3, 3), activation = 'relu'))



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(num_classes, activation = 'softmax'))



model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_set, epochs=80, verbose=1, validation_data = validation_set, shuffle=True)
import matplotlib.pyplot as plt





# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")
'''

from keras.models import model_from_json

# load json and create model

json_file = open('/kaggle/working/model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("/kaggle/working/model.h5")

print("Loaded model from disk")

'''
test_loss , test_accuracy = model.evaluate(test_set)
print("Test accuracy: "+ str(test_accuracy*100), "\nTest_loss: " + str(test_loss*100))
from skimage.transform import resize

from PIL import Image

import numpy as np

im = Image.open(path+'/Test/character_28_la/11454.png')

im = np.array(im)

im = resize(im , (1,64,64,3))

y_pred = model.predict(im)
yy = np.argmax(y_pred)

print(yy)
print(key_list[int(yy)])