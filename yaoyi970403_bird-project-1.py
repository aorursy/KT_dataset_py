from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers, models, Model, Sequential

import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

import tensorflow as tf

import json

import os
im_height = 224

im_width = 224

batch_size = 128

epochs = 15
# create direction for saving weights

if not os.path.exists("save_weights"):

    os.makedirs("save_weights")
image_path = "../input/100-bird-species/"

train_dir = image_path + "train"

validation_dir = image_path + "valid"

test_dir = image_path + "test"



train_image_generator = ImageDataGenerator( rescale=1./255, 

                                            rotation_range=40, 

                                            width_shift_range=0.2,

                                            height_shift_range=0.2, 

                                           shear_range=0.2,

                                            zoom_range=0.2,

                                            horizontal_flip=True, 

                                            fill_mode='nearest')



train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,

                                                           batch_size=batch_size,

                                                           shuffle=True,

                                                           target_size=(im_height, im_width),

                                                           class_mode='categorical')

    

total_train = train_data_gen.n





validation_image_generator = ImageDataGenerator(rescale=1./255)



val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,

                                                              batch_size=batch_size,

                                                              shuffle=False,

                                                              target_size=(im_height, im_width),

                                                              class_mode='categorical')

    

total_val = val_data_gen.n







test_image_generator = ImageDataGenerator(rescale=1./255)



test_data_gen = test_image_generator.flow_from_directory( directory=test_dir,

                                                          batch_size=batch_size,

                                                          shuffle=False,

                                                          target_size=(im_height, im_width),

                                                          class_mode='categorical')

    

total_test = test_data_gen.n
covn_base = tf.keras.applications.NASNetLarge(weights='imagenet', include_top = False)

covn_base.trainable = True



print(len(covn_base.layers))



for layers in covn_base.layers[:-30]:

    layers.trainable = False



model = tf.keras.Sequential([

        covn_base,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(200, activation='softmax')

    ])

model.summary()     

model.compile(

    optimizer=tf.keras.optimizers.Adam(),

    loss = 'categorical_crossentropy',

    metrics=['accuracy']

)
def lrfn(epoch):

    LR_START = 0.00001

    LR_MAX = 0.0004

    LR_MIN = 0.00001

    LR_RAMPUP_EPOCHS = 5

    LR_SUSTAIN_EPOCHS = 0

    LR_EXP_DECAY = .8

    

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr



rng = [i for i in range(epochs)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)





checkpoint = ModelCheckpoint(

                                filepath='./save_weights/myNASNetLarge.ckpt',

                                monitor='val_acc', 

                                save_weights_only=False, 

                                save_best_only=True, 

                                mode='auto',

                                period=1

                            )



history = model.fit(x=train_data_gen,

                    steps_per_epoch=total_train // batch_size,

                    epochs=epochs,

                    validation_data=val_data_gen,

                    validation_steps=total_val // batch_size,

                    callbacks=[checkpoint, lr_schedule])
model.save_weights('./save_weights/myNASNetLarge.ckpt',save_format='tf')
# plot loss and accuracy image

history_dict = history.history

train_loss = history_dict["loss"]

train_accuracy = history_dict["accuracy"]

val_loss = history_dict["val_loss"]

val_accuracy = history_dict["val_accuracy"]



# figure 1

plt.figure()

plt.plot(range(epochs), train_loss, label='train_loss')

plt.plot(range(epochs), val_loss, label='val_loss')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('loss')



# figure 2

plt.figure()

plt.plot(range(epochs), train_accuracy, label='train_accuracy')

plt.plot(range(epochs), val_accuracy, label='val_accuracy')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.show()
test_data_gen = test_image_generator.flow_from_directory( directory=test_dir,

                                                          target_size=(im_height, im_width))



total_test = test_data_gen.n
scores = model.evaluate(test_data_gen, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])
from PIL import Image

import numpy as np  

#获取数据集的类别编码

class_indices = train_data_gen.class_indices 

#将编码和对应的类别存入字典

inverse_dict = dict((val, key) for key, val in class_indices.items()) 

#加载测试图片

img = Image.open("../input/100-bird-species/test/AFRICAN FIREFINCH/1.jpg")

# 将图片resize到224x224大小

img = img.resize((im_width, im_height))

# 归一化

img1 = np.array(img) / 255.

# 将图片增加一个维度，目的是匹配网络模型

img1 = (np.expand_dims(img1, 0))

#将预测结果转化为概率值

result = np.squeeze(model.predict(img1))

predict_class = np.argmax(result)

#print(inverse_dict[int(predict_class)],result[predict_class])

#将预测的结果打印在图片上面

plt.title([inverse_dict[int(predict_class)],result[predict_class]])

#显示图片

plt.imshow(img)