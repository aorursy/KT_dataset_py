import glob



# 了解数据集的组成



train_files = glob.glob('/kaggle/input/dogs-vs-cats/train/train/*')

test_files = glob.glob('/kaggle/input/dogs-vs-cats/test1/test1/*')



train_cat_files = [file_name for file_name in train_files if 'cat' in file_name]

train_dog_files = [file_name for file_name in train_files if 'dog' in file_name]



print('train samples of cat:', len(train_cat_files))

print('train samples of dog:', len(train_dog_files))

print( train_dog_files[0])

import numpy as np

from random import shuffle

from keras.preprocessing.image import load_img, img_to_array

from sklearn.preprocessing import LabelEncoder



# 从猫训练数据中随机抽取1500张训练样本

cat_train = list(np.random.choice(train_cat_files, size=1500, replace=False))



# 从狗训练数据中随机抽取1500张训练样本

dog_train = list(np.random.choice(train_dog_files, size=1500, replace=False))



# 从猫训练数据中剔除已经抽取的训练样本

train_cat_files = list(set(train_cat_files) - set(cat_train))



# 从狗训练数据中剔除已经抽取的训练样本

train_dog_files = list(set(train_dog_files) - set(dog_train))



# 从猫训练数据中随机抽取500张校验样本

cat_val = list(np.random.choice(train_cat_files, size=500, replace=False))



# 从狗训练数据中随机抽取500张校验样本

dog_val = list(np.random.choice(train_dog_files, size=500, replace=False))



# 从猫训练数据中剔除已经抽取的校验样本

train_cat_files = list(set(train_cat_files) - set(cat_val))



# 从狗训练数据中剔除已经抽取的校验样本

train_dog_files = list(set(train_dog_files) - set(dog_val))



# 从猫训练数据中随机抽取500张测试样本

cat_test = list(np.random.choice(train_cat_files, size=500, replace=False))



# 从狗训练数据中随机抽取500张测试样本

dog_test = list(np.random.choice(train_dog_files, size=500, replace=False))



# 合并猫狗训练集

train_files = cat_train + dog_train

# 合并猫狗校验集

val_files = cat_val + dog_val

# 合并猫狗测试集

test_files = cat_test + dog_test



# 随机化猫狗训练集

shuffle(train_files)



# 样本尺寸

IMG_DIM = (150, 150)

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

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,

                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 

                                   horizontal_flip=True, fill_mode='nearest')



val_datagen = ImageDataGenerator(rescale=1./255)



img_id = 3

cat_generator = train_datagen.flow(x_train[img_id:img_id+1], train_labels[img_id:img_id+1],

                                   batch_size=1)

cat = [next(cat_generator) for i in range(0,5)]

fig, ax = plt.subplots(1,5, figsize=(16, 6))

print('Labels:', [item[1][0] for item in cat])

l = [ax[i].imshow(cat[i][0][0]) for i in range(0,5)]
img_id = 5

cat_generator = train_datagen.flow(x_train[img_id:img_id+1], train_labels[img_id:img_id+1],

                                   batch_size=1)

cat = [next(cat_generator) for i in range(0,5)]

fig, ax = plt.subplots(1,5, figsize=(16, 6))

print('Labels:', [item[1][0] for item in cat])

l = [ax[i].imshow(cat[i][0][0]) for i in range(0,5)]
train_generator = train_datagen.flow(x_train, y_train, batch_size=30)

val_generator = val_datagen.flow(x_val, y_val, batch_size=20)

input_shape = (150, 150, 3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.models import Sequential

from keras import optimizers



batch_size = 30

num_classes = 2

epochs = 30

input_shape = (150, 150, 3)



model = Sequential(name='Basic cnn model with Augmentation')



# 第一卷积层

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))



# 第二卷积层

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



# 第三卷积层

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



# 第四卷积层

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



# 扁平化层

model.add(Flatten())



# 第一全连接层

model.add(Dense(512, activation='relu'))



# 正则化，随机裁剪30%

model.add(Dropout(0.3))



# 第二全连接层

model.add(Dense(512, activation='relu'))



# 正则化，随机裁剪30%

model.add(Dropout(0.3))







# 第三全连接层

model.add(Dense(1, activation='sigmoid'))



# 编译模型，指定损失计算使用binary_crossentropy，优化器使用RMSprop，模型性能度量使用accuracy

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])



model.summary()

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,

                              validation_data=val_generator, validation_steps=50, 

                              verbose=1)
import matplotlib.pyplot as plt



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t = f.suptitle('Basic CNN Performance', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = list(range(1,101))

ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')

ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, 31, 5))

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch #')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, history.history['loss'], label='Train Loss')

ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, 31, 5))

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch #')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")
model.save('cats_dogs_basic_cnn_re_aug.h5')