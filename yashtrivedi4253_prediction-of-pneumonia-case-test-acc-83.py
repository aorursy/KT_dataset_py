
import os
import numpy as np
import matplotlib.pyplot as plt

import cv2

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
base_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/"

train_dir = os.path.join(base_dir + "train")
test_dir = os.path.join(base_dir + "test")
val_dir = os.path.join(base_dir + "val")
train_normal = os.path.join(train_dir,"NORMAL")
train_pneumonia = os.path.join(train_dir , "PNEUMONIA")

test_normal = os.path.join(test_dir , "NORMAL")
test_pneumonia = os.path.join(test_dir , "PNEUMONIA")

val_normal = os.path.join(val_dir , "NORMAL")
val_pneumonia = os.path.join(val_dir , "PNEUMONIA")
num_train_normal = len(os.listdir(train_normal))
num_train_pneumonia = len(os.listdir(train_pneumonia))

num_test_normal = len(os.listdir(test_normal))
num_test_pneumonia = len(os.listdir(test_pneumonia))

num_val_normal = len(os.listdir(val_normal))
num_val_pneumonia = len(os.listdir(val_pneumonia))

print("num_train_normal: ",num_train_normal)
print("num_train_pneumonia: ",num_train_pneumonia)
print()
print("num_test_normal: ",num_test_normal)
print("num_test_pneumonia: ",num_test_pneumonia)
print()
print("num_val_normal: ",num_val_normal)
print("num_val_pneumonia: ",num_val_pneumonia)

total_train_data = num_train_normal + num_train_pneumonia
total_test_data = num_test_normal + num_test_pneumonia
total_val_data = num_val_normal + num_val_pneumonia

print("Total train dataset is: ", total_train_data)
print("Total test dataset is: ", total_test_data)
print("Total val dataset is: ", total_val_data)

batch_size = 32
img_shape = 150
train_image_generator = ImageDataGenerator(rotation_range=0,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rescale = 1./255)

test_image_generator = ImageDataGenerator(rescale = 1./255)

val_image_generator = ImageDataGenerator(rescale = 1./255)
train_data_gen = train_image_generator.flow_from_directory(train_dir,target_size =(img_shape,img_shape),batch_size=batch_size, shuffle=True,class_mode="binary")

test_data_gen = test_image_generator.flow_from_directory(test_dir,
    target_size=(img_shape,img_shape),
    class_mode='binary')

val_data_gen = val_image_generator.flow_from_directory(directory=val_dir,
    target_size=(img_shape,img_shape),
    class_mode='binary')


model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1, 1),padding='same',activation="relu",input_shape=(img_shape,img_shape,3)))
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1, 1),padding='same',activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1, 1),padding='same',activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1, 1),padding='same',activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=132,kernel_size=(3,3),strides=(1, 1),padding='same',activation="relu"))
model.add(Conv2D(filters=132,kernel_size=(3,3),strides=(1, 1),padding='same',activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
          
model.add(Dense(units=1024, activation = "relu"))
model.add(Dense(units=128, activation = "relu"))
model.add(Dense(units=1, activation = "sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["acc"])
model.summary()
h = model.fit_generator(generator = train_data_gen,
    steps_per_epoch=int( np.ceil(total_train_data) / float(batch_size)),
    epochs=8,
    verbose=1,
    validation_data=val_data_gen,
    validation_steps= int( np.ceil(total_val_data) / float(batch_size)))
plt.plot(h.history["acc"])
plt.plot(h.history["val_acc"])
plt.xlabel("Epochs")
plt.title("Accuracy")
plt.legend(["acc","val_acc"])
plt.show()
plt.plot(h.history["loss"])
plt.plot(h.history["val_loss"])
plt.xlabel("Epochs")
plt.title("Loss")
plt.legend(["loss","val_loss"])

plt.show()
score = model.evaluate_generator(test_data_gen,verbose=1)
print("Test loss is: ", score[0])
print("Test accuracy is: ", str(round(score[1]*100,3)) + "%")
pred = model.predict_generator(test_data_gen)
y_pred = model.predict_generator(test_data_gen,100)
y_pred = np.argmax(y_pred,axis=1)
img_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg"


img1 = cv2.imread(img_path)
img = img1/255.

img = img.astype(np.float)
img = cv2.resize(img,(150,150))
img = img.reshape(1,150,150,3)
iimmgg = model.predict_classes(img)
print(iimmgg)
plt.imshow(img1)
plt.show()
img2_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0015-0001.jpeg"
img21 = cv2.imread(img2_path)
img2 = img21/255

img2 = img2.astype(np.float)
img2 = cv2.resize(img2,(150,150))
img2 = img2.reshape(1,150,150,3)
iimmgg2 = model.predict_classes(img2)
print(iimmgg2)
plt.imshow(img21)
plt.show()
