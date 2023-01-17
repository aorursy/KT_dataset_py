import numpy as np

import cv2

from keras.callbacks import ModelCheckpoint,EarlyStopping

from keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout,SpatialDropout2D

from keras.models  import Sequential

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

import random,os,glob

import matplotlib.pyplot as plt
dir_path = '../input/garbage classification/Garbage classification'
img_list = glob.glob(os.path.join(dir_path, '*/*.jpg'))
len(img_list)
train=ImageDataGenerator(horizontal_flip=True,

                         vertical_flip=True,

                         validation_split=0.1,

                         rescale=1./255,

                         shear_range = 0.1,

                         zoom_range = 0.1,

                         width_shift_range = 0.1,

                         height_shift_range = 0.1,)



test=ImageDataGenerator(rescale=1/255,

                        validation_split=0.1)



train_generator=train.flow_from_directory(dir_path,

                                          target_size=(300,300),

                                          batch_size=32,

                                          class_mode='categorical',

                                          subset='training')



test_generator=test.flow_from_directory(dir_path,

                                        target_size=(300,300),

                                        batch_size=32,

                                        class_mode='categorical',

                                        subset='validation')



labels = (train_generator.class_indices)

print(labels)



labels = dict((v,k) for k,v in labels.items())

print(labels)
for image_batch, label_batch in train_generator:

  break

image_batch.shape, label_batch.shape
print (train_generator.class_indices)



Labels = '\n'.join(sorted(train_generator.class_indices.keys()))



with open('labels.txt', 'w') as f:

  f.write(Labels)

model=Sequential()

#Convolution blocks



model.add(Conv2D(32,(3,3), padding='same',input_shape=(300,300,3),activation='relu'))

model.add(MaxPooling2D(pool_size=2)) 

#model.add(SpatialDropout2D(0.5)) # No accuracy



model.add(Conv2D(64,(3,3), padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=2)) 

#model.add(SpatialDropout2D(0.5))



model.add(Conv2D(32,(3,3), padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=2)) 



#Classification layers

model.add(Flatten())



model.add(Dense(64,activation='relu'))

#model.add(SpatialDropout2D(0.5))

model.add(Dropout(0.2))

model.add(Dense(32,activation='relu'))



model.add(Dropout(0.2))

model.add(Dense(6,activation='softmax'))



filepath="trained_model.h5"

checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint1]



model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc']) # RMS PROP - No accuracy



#es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

history = model.fit_generator(train_generator,

                              epochs=100,

                              steps_per_epoch=2276//32,

                              validation_data=test_generator,

                              validation_steps=251//32,

                              workers = 4,

                              callbacks=callbacks_list) 

#41 epoch - 75% #73- 76.9%

#78 epoch - 80%
from keras.preprocessing import image



img_path = '../input/garbage classification/Garbage classification/plastic/plastic75.jpg'



img = image.load_img(img_path, target_size=(300, 300))

img = image.img_to_array(img, dtype=np.uint8)

img=np.array(img)/255.0



plt.title("Loaded Image")

plt.axis('off')

plt.imshow(img.squeeze())



p=model.predict(img[np.newaxis, ...])



#print("Predicted shape",p.shape)

print("Maximum Probability: ",np.max(p[0], axis=-1))

predicted_class = labels[np.argmax(p[0], axis=-1)]

print("Classified:",predicted_class)



classes=[]

prob=[]

print("\n-------------------Individual Probability--------------------------------\n")



for i,j in enumerate (p[0],0):

    print(labels[i].upper(),':',round(j*100,2),'%')

    classes.append(labels[i])

    prob.append(round(j*100,2))

    

def plot_bar_x():

    # this is for plotting purpose

    index = np.arange(len(classes))

    plt.bar(index, prob)

    plt.xlabel('Labels', fontsize=12)

    plt.ylabel('Probability', fontsize=12)

    plt.xticks(index, classes, fontsize=12, rotation=20)

    plt.title('Probability for loaded image')

    plt.show()

plot_bar_x()
acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



# ________________ Graph 1 -------------------------



plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([min(plt.ylim()),1])

plt.title('Training and Validation Accuracy')



# ________________ Graph 2 -------------------------



plt.subplot(2, 1, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.ylim([0,max(plt.ylim())])

plt.title('Training and Validation Loss')

plt.show()
import tensorflow as tf

import keras

file = "Garbage.h5"

keras.models.save_model(model,file)

converter = tf.lite.TFLiteConverter.from_keras_model_file(file)

tflite_model=converter.convert()

open("garbage.tflite",'wb').write(tflite_model)
from IPython.display import FileLinks

FileLinks('.')