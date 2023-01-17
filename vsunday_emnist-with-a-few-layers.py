import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
train_data_path = '../input/emnist-balanced-train.csv'
test_data_path = '../input/emnist-balanced-test.csv'

num_classes = 47
img_size = 28

def img_label_load(data_path, num_classes=None):
    data = pd.read_csv(data_path, header=None)
    data_rows = len(data)
    if not num_classes:
        num_classes = len(data[0].unique())
    img_size = int(np.sqrt(len(data.iloc[0][1:])))
    imgs = np.transpose(data.values[:,1:].reshape(data_rows, img_size, img_size, 1), axes=[0,2,1,3]) # img_size * img_size arrays
    labels = to_categorical(data.values[:,0], num_classes) # one-hot encoding vectors
    
    return imgs/255, labels
model = Sequential()

model.add(Conv2D(filters=60, kernel_size=(2,2), strides=2, activation='relu', input_shape=(img_size,img_size,1)))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.5))

model.add(Conv2D(filters=60, kernel_size=(3,3) , strides=2, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.5))

model.add(Conv2D(filters=60, kernel_size=(3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(filters=60, kernel_size=(3,3), activation='relu'))

model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
model.summary()
data_generator = ImageDataGenerator(validation_split=.2)
data_generator_with_aug = ImageDataGenerator(validation_split=.2,
                                            width_shift_range=.2, height_shift_range=.2,
                                            rotation_range=60, zoom_range=.2, shear_range=.3)
X, y = img_label_load(train_data_path)
training_data_generator = data_generator_with_aug.flow(X, y, subset='training')
validation_data_generator = data_generator.flow(X, y, subset='validation')
history = model.fit_generator(training_data_generator, steps_per_epoch=500, epochs=100, validation_data=validation_data_generator)
test_X, test_y = img_label_load(test_data_path)
test_data_geneartor = data_generator.flow(X, y)

model.evaluate_generator(test_data_geneartor)
with open('model.json', 'w') as f:
    f.write(model.to_json())
model.save_weights('./model.h5')
!ls -lh
import matplotlib.pyplot as plt

print(history.history.keys())

# accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()