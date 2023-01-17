# source : https://www.kaggle.com/amitkrjha/plant-disease-detection-using-vgg16



import time

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from tensorflow.keras.preprocessing import image
input_path = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/"



train_datagen = ImageDataGenerator(

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    width_shift_range=0.2,

    height_shift_range=0.2,

    fill_mode='nearest',

    horizontal_flip=True,

    preprocessing_function=preprocess_input

)



validation_datagen = ImageDataGenerator(

    rescale=1./255,

    preprocessing_function=preprocess_input

)



batch_size = 128

train_set = train_datagen.flow_from_directory(

    input_path + 'train',                                          

    batch_size=batch_size,

    class_mode='categorical',

    target_size=(224,224)

)



validation_set = validation_datagen.flow_from_directory(

    input_path + 'valid',

    shuffle=False,

    batch_size=batch_size,

    class_mode='categorical',

    target_size=(224,224)

)
base_model=VGG16(weights='imagenet', include_top=False,input_shape=(224,224,3))

base_model.trainable=False



model=Sequential()

model.add(base_model)

model.add(Flatten())

model.add(Dense(38,activation='softmax'))

model.summary()



model.compile(optimizer='adam',

                   loss='categorical_crossentropy',

                   metrics=['accuracy'])
train_num = train_set.samples

valid_num = validation_set.samples
t=time.time()

history = model.fit_generator(train_set,

                              steps_per_epoch=train_num//batch_size,

                              validation_data=validation_set,

                              epochs=5,

                              validation_steps=valid_num//batch_size

)

print('Training time: %s' % (t - time.time()))
class_dict = train_set.class_indices

print(class_dict)
li = list(class_dict.keys())

print(li)
import matplotlib.pyplot as plt

import seaborn as sns



sns.set()



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)



#accuracy plot

plt.plot(epochs, acc, color='green', label='Training Accuracy')

plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')

plt.title('Training and Validation Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend()



plt.figure()

#loss plot

plt.plot(epochs, loss, color='pink', label='Training Loss')

plt.plot(epochs, val_loss, color='red', label='Validation Loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()



plt.show()
image_path = "../input/new-plant-diseases-dataset/test/test/TomatoEarlyBlight1.JPG"

new_img = image.load_img(image_path, target_size=(224, 224))

img = image.img_to_array(new_img)

img = np.expand_dims(img, axis=0)

img = img/255



print("Following is our prediction:")

prediction = model.predict(img)

# decode the results into a list of tuples (class, description, probability)

# (one such list for each sample in the batch)

d = prediction.flatten()

j = d.max()

for index,item in enumerate(d):

    if item == j:

        class_name = li[index]



#ploting image with predicted class name        

plt.figure(figsize = (4,4))

plt.imshow(new_img)

plt.axis('off')

plt.title(class_name)

plt.show()
filepath="Mymodel.h5"

model.save(filepath)
import os

from IPython.display import FileLink

os.chdir(r'/kaggle/working')

FileLink(r'Mymodel.h5')