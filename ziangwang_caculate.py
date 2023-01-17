from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

import matplotlib.pyplot as plt

import numpy as np

import os



fnames = [os.path.join("/kaggle/input/garbage-34/garbage_34/cans", fnames) for fnames in os.listdir("/kaggle/input/garbage-34/garbage_34/cans")]



datagen = ImageDataGenerator(rescale=1./255, 

                             rotation_range=30, 

                             zoom_range=0.3, 

                             width_shift_range=0.2, 

                             height_shift_range=0.2, 

                             horizontal_flip=True)



img_path = fnames[20]

img = load_img(img_path,target_size=(299, 299))

img = np.expand_dims(img, axis=0)



i = 0



for batch in datagen.flow(img, batch_size = 1):

    plt.figure(i)

    imgplot = plt.imshow(img_to_array(batch[0]))

    i += 1

    if i % 3 == 0:

        break
from keras.preprocessing.image import ImageDataGenerator



data_path = '/kaggle/input/garbage-34/garbage_34/'



datagen = ImageDataGenerator(rescale=1./255, 

                             rotation_range=30, 

                             zoom_range=0.3, 

                             width_shift_range=0.2, 

                             height_shift_range=0.2, 

                             horizontal_flip=True, 

                             validation_split=0.2) # 数据集的80%用于训练，20%用于验证



training_generator = datagen.flow_from_directory(data_path, 

                                                 target_size=(299, 299), 

                                                 batch_size=32, 

                                                 class_mode='categorical', 

                                                 shuffle=True, 

                                                 subset='training')



validation_generator = datagen.flow_from_directory(data_path, 

                                                   target_size=(299, 299), 

                                                   batch_size=32, 

                                                   class_mode='categorical', 

                                                   shuffle=True, 

                                                   subset='validation')



print('\n',training_generator.class_indices)   # class和index对应关系以字典的形式输出
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation, Flatten

from keras.models import Sequential



model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',  # 这里添加的第一层为包含32个3*3大小的卷积核的卷积层。

                 input_shape=(299, 299, 3)))  # 还记得吗？模型的第一层需要指定输入大小，这里我们假设输入是299*299的RGB图像。

model.add(Activation('relu'))  # 每添加一层，都要为这层添加激活函数。

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))  # 训练期间的每次更新中将输入单元的一部分随机设置为0，这有助于防止过拟合。



model.add(Flatten())  # 下面要添加全连接层了，所以要把图片展平成全连接层的输入。

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(34))  # 这里是模型的最后一层，在图像分类问题中Dense的参数是分类的数目。

model.add(Activation('softmax'))
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Activation, Flatten, Input, Dropout

from keras.models import Sequential, Model

from keras.applications import InceptionV3

from PIL import ImageFile

import warnings



# 因为数据集是从网站上下载的，会存在图片损坏的情况

# 使用这句命令可以忽略由于图片损坏导致模型训练出现错误而停止训练

ImageFile.LOAD_TRUNCATED_IMAGES = True



# 笔者是个强迫症患者，运行python经常出现各种警告，看着不太舒服，所以用这句命令忽略这些警告

warnings.filterwarnings("ignore")



# InceptionV3不包含顶层的权重的路径

weight = '/kaggle/input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'



# InceptionV3的默认输入为(299, 299)，所以我们这里定义模型输入为299*299的RGB图像

input_tensor = Input(shape=(299, 299, 3))



# 因为要把模型运用在我们自己的数据集上，所以不要包含InceptionV3的顶层。

# 我们提供了模型的权重，所以weights要设为None，如果设为‘imagenet’，会自动下载权重，如果不翻墙的话基本是不会下载成功的。

base_model = InceptionV3(input_tensor=input_tensor, weights=None, include_top=False)



# 加载模型用.load_model()方法，但是我们提供的是InceptionV3的权重，所以要用.load_weights()方法加载权重。

base_model.load_weights(weight)



# 主要是用来解决全连接的问题，其主要是是将最后一层的特征图进行整张图的一个均值池化，形成一个特征点，将这些特征点组成最后的特征向量进行softmax中进行计算。

x = base_model.output

x = GlobalAveragePooling2D()(x)



# 添加全连接层

x = Dense(1024, activation='relu')(x)

x = Dropout(0.5)(x)



# 这里是模型的最后一层Dense的参数是我们垃圾分类的数量。

predictions = Dense(34, activation='softmax')(x)



# 我们在InceptionV3后又添加了三层，所以用Model来把新的模型进行封装。

model = Model(inputs=input_tensor, outputs=predictions)

model.summary()
from keras.utils import plot_model

plot_model(model, to_file='model.png')
# 冻结所有卷积层，只训练顶层，因为它是随机初始化的

for layer in base_model.layers:

    layer.trainable = False
model.compile(optimizer='rmsprop', 

              loss='categorical_crossentropy', 

              metrics=['acc'])
# 在数据集上训练少量的epochs

history = model.fit_generator(training_generator, 

                              shuffle=True, 

                              epochs=5, 

                              verbose=1, 

                              validation_data=validation_generator)
import matplotlib.pyplot as plt



# 可视化训练集和验证集的正确率

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# 可视化训练集和验证集的损失值

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# 我们可以看到每一层的名字，考虑下一步要冻结哪些层

for i, layer in enumerate(base_model.layers):

    print(i, layer.name)
# 我们选择冻结前两个inception模块

# 解冻剩下的所有层

for layer in model.layers[:249]:

    layer.trainable = False

for layer in model.layers[249:]:

    layer.trainable = True
# 我们需要重新编译模型以使这些修改生效

from keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), 

              loss='categorical_crossentropy', 

              metrics=['acc'])
# 我们再次训练模型

history = model.fit_generator(training_generator, 

                              shuffle=True, 

                              epochs=15, 

                              verbose=1, 

                              validation_data=validation_generator)
# 可视化训练集和验证集的正确率

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# 可视化训练集和验证集的损失值

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
model.save('garbage_34.h5')
from keras.preprocessing.image import img_to_array, load_img

import matplotlib.pyplot as plt



# 图片路径

img = '/kaggle/input/zheshiceshi/IMG_20191024_212905.jpg'



# 加载图像

img = load_img(img, target_size=(299, 299))



# 看一看我们自己拍摄的图片

plt.imshow(img)



# 归一化

img = img_to_array(img) / 255.0



# 扩充维度，使图像适合卷积层的输入

img = np.expand_dims(img, axis=0)



# 使用模型来预测图像的类别，得到图片符合每一类的概率

predict = model.predict(img)



# 取最大的概率所对应的下标

predict = np.argmax(predict, axis=1)



# 在介绍ImageDataGenerator时，我们已经输出了类别和标号的关系，所以可以通过输出的标号去匹配这张图像预测为哪一类别

print(predict)