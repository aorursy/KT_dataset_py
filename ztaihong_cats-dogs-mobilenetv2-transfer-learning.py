import glob



# 了解数据集的组成



train_files = glob.glob('/kaggle/input/dogs-vs-cats/train/train/*')

test_files = glob.glob('/kaggle/input/dogs-vs-cats/test1/test1/*')



train_cat_files = [file_name for file_name in train_files if 'cat' in file_name]

train_dog_files = [file_name for file_name in train_files if 'dog' in file_name]



print('train samples of cat:', len(train_cat_files))

print('train samples of dog:', len(train_dog_files))
import numpy as np

from random import shuffle

import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.preprocessing import LabelEncoder



# 从猫训练数据中随机抽取1500张训练样本

cat_train = list(np.random.choice(train_cat_files, size=5000, replace=False))



# 从狗训练数据中随机抽取1500张训练样本

dog_train = list(np.random.choice(train_dog_files, size=5000, replace=False))



# 从猫训练数据中剔除已经抽取的训练样本

train_cat_files = list(set(train_cat_files) - set(cat_train))



# 从狗训练数据中剔除已经抽取的训练样本

train_dog_files = list(set(train_dog_files) - set(dog_train))



# 从猫训练数据中随机抽取500张校验样本

cat_val = list(np.random.choice(train_cat_files, size=1000, replace=False))



# 从狗训练数据中随机抽取500张校验样本

dog_val = list(np.random.choice(train_dog_files, size=1000, replace=False))



# 从猫训练数据中剔除已经抽取的校验样本

train_cat_files = list(set(train_cat_files) - set(cat_val))



# 从狗训练数据中剔除已经抽取的校验样本

train_dog_files = list(set(train_dog_files) - set(dog_val))



# 从猫训练数据中随机抽取500张测试样本

cat_test = list(np.random.choice(train_cat_files, size=1000, replace=False))



# 从狗训练数据中随机抽取500张测试样本

dog_test = list(np.random.choice(train_dog_files, size=1000, replace=False))



# 合并猫狗训练集

train_files = cat_train + dog_train

# 合并猫狗校验集

val_files = cat_val + dog_val

# 合并猫狗测试集

test_files = cat_test + dog_test



# 随机化猫狗训练集

shuffle(train_files)



# 样本尺寸

IMG_DIM = (160, 160)

# 从磁盘加载训练集

x_train = np.array([img_to_array(load_img(image_file, target_size=IMG_DIM)) for image_file in train_files])

# 从磁盘加载校验集

x_val = np.array([img_to_array(load_img(image_file, target_size=IMG_DIM)) for image_file in val_files])

# 从磁盘加载测试集

x_test = np.array([img_to_array(load_img(image_file, target_size=IMG_DIM)) for image_file in test_files])



# 将训练集列表转换为numpy矩阵

x_train = np.array(x_train)

# 将校验集列表转换为numpy矩阵

x_val = np.array(x_val)

# 将测试集列表转换为numpy矩阵

x_test = np.array(x_test)



# 标签编码

train_labels = [fn.split('/')[-1].split('.')[0].strip() for fn in train_files]

val_labels = [fn.split('/')[-1].split('.')[0].strip() for fn in val_files]

test_labels = [fn.split('/')[-1].split('.')[0].strip() for fn in test_files]

le = LabelEncoder()

le.fit(train_labels)

y_train = le.transform(train_labels)

y_val = le.transform(val_labels)

y_test = le.transform(test_labels)





print('x_train shape:', x_train.shape)

print('y_train shape:', y_train.shape)

print('x_validate shape:', x_val.shape)

print('y_validate shape:', y_val.shape)

print('x_test shape:', x_test.shape)

print('y_test shape:', y_test.shape)
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, 

                                   zoom_range=0.3, 

                                   rotation_range=50,

                                   width_shift_range=0.2, 

                                   height_shift_range=0.2, 

                                   shear_range=0.2, 

                                   horizontal_flip=True, 

                                   fill_mode='nearest')



val_datagen = ImageDataGenerator(rescale=1./255)



img_id = 3

generator = train_datagen.flow(x_train[img_id:img_id+1], train_labels[img_id:img_id+1])

images = [next(generator) for i in range(0,5)]

fig, ax = plt.subplots(1,5, figsize=(16, 6))

print('Labels:', [item[1][0] for item in images])

l = [ax[i].imshow(images[i][0][0]) for i in range(0,5)]
from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import Flatten

from tensorflow.keras.models import Model

import pandas as pd





input_shape = (160, 160, 3)

mobilenet_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

#output = mobilenet.layers[-1].output

#output = Flatten()(output)

#mobilenet_model = Model(mobilenet.input, output)



mobilenet_model.trainable = True

#fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer

#for layer in mobilenet_model.layers[:fine_tune_at]:

#    layer.trainable =  False



mobilenet_model.summary()
layers = [(layer, layer.name, layer.trainable) for layer in mobilenet_model.layers]

pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])  
print(x_train.shape[0])
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, GlobalAveragePooling2D

from tensorflow.keras.models import Sequential

from tensorflow.keras import optimizers



BATCH_SIZE = 32

TRAIN_SAMPLE_NUMBER = x_train.shape[0]

VAL_SAMPLE_NUMBER = x_val.shape[0]

steps_per_epoch = TRAIN_SAMPLE_NUMBER // BATCH_SIZE

epochs = 20

validation_steps = VAL_SAMPLE_NUMBER // BATCH_SIZE



train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

val_generator = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE)



model = Sequential()

model.add(mobilenet_model)

model.add(GlobalAveragePooling2D())

'''

model.add(Dense(512, activation='relu', input_dim=input_shape))

model.add(Dropout(0.3))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

'''

model.add(Dense(units=2, activation='softmax'))



model.compile(loss='sparse_categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-5),

              metrics=['accuracy'])

              

history = model.fit_generator(train_generator, 

                              steps_per_epoch=steps_per_epoch, 

                              epochs=epochs,

                              validation_data=val_generator, 

                              validation_steps=validation_steps, 

                              verbose=1)    
import matplotlib.pyplot as plt



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t = f.suptitle('Pre-trained MobileNetV2 Transfer Learn with Fine-Tuning & Image Augmentation Performance ', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = list(range(1,21))

ax1.plot(epoch_list, history.history['acc'], label='Train Accuracy')

ax1.plot(epoch_list, history.history['val_acc'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, 21, 1))

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch #')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, history.history['loss'], label='Train Loss')

ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, 21, 1))

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch #')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")
model.save('Cats_Dogs_MobileNetV2_Fine_Tuning_Transfer_Learning.h5')