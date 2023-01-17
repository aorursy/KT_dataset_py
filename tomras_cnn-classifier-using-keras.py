import json

import numpy as np

from matplotlib import pyplot as plt

from keras.utils.np_utils import to_categorical



from keras.models import Sequential

from keras.layers.core import Flatten, Dense, Dropout, Lambda

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.optimizers import Adam, SGD
f = open(r'../input/shipsnet.json')

dataset = json.load(f)

f.close()
data = np.array(dataset['data']).astype('uint8')

labels = np.array(dataset['labels']).astype('uint8')
print(data.shape)

print(labels.shape)
x = data / 255.

x = x.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])



print(x.shape)
y = to_categorical(labels, num_classes=2)

print(y.shape)
img_id_to_check = np.random.randint(0, x.shape[0] - 1)

im = x[img_id_to_check]



print(img_id_to_check)

print(y[img_id_to_check])



plt.imshow(im)

plt.show()
model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", input_shape=(80, 80, 3), activation='relu'))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.2))



# model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))

# model.add(Conv2D(128, (3, 3), activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), 

              metrics=['accuracy'])
history = model.fit(x, y, batch_size=32, epochs=20, validation_split=0.2)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()