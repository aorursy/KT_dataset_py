import numpy as np

import glob

import os

import scipy.misc

import gc





# 加载数据

def load_data(dataset_path):

    print('正在加载数据集......')

    print('##############################################################')

    train_image = []

    train_label = []

    validate_image = []

    validate_label = []

    for dir_path in glob.glob(dataset_path + "/Training/*"):

        img_label = dir_path.split("/")[-1]

        for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):

            img = scipy.misc.imread(img_path, mode='RGB')

            train_image.append(img)

            train_label.append(img_label)



    x_training = np.array(train_image)

    del train_image

    gc.collect()

    x_training = x_training / 255.0

    label_training = np.array(train_label)

    label_to_id = {v: k for k, v in enumerate(np.unique(label_training))}

    class_dictionary = {v: k for k, v in label_to_id.items()}

    class_no = len(class_dictionary)

    y_training = np.array([label_to_id[i] for i in label_training])



    del train_label

    gc.collect()

    

    for dir_path in glob.glob(dataset_path + "/Validation/*"):

        image_label = dir_path.split("/")[-1]

        for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):

            img = scipy.misc.imread(img_path, mode='RGB')

            validate_image.append(img)

            validate_label.append(image_label)

    x_validation = np.array(validate_image)

    del validate_image

    gc.collect()

    

    x_validation = x_validation / 255.0

    label_validate = np.array(validate_label)

    y_validation = np.array([label_to_id[i] for i in label_validate])

    

    del label_validate

    gc.collect()

    

    print('x_train dimension:', x_training.shape)

    print('y_train dimension:', y_training.shape)

    print('x_validate dimension:', x_validation.shape)

    print('y_validate dimension:', y_validation.shape)

    print('class_dict', class_dictionary)

    print('class_number', class_no)

    print('##############################################################')

    print('数据集加载完毕！！！')

    return (x_training, y_training), (x_validation, y_validation), class_dictionary, class_no
import tensorflow as tf





# 数据增强

def augment_image(image):



    # 将RGB格式转换为灰度图像

    gray = tf.image.rgb_to_grayscale(image)



    # 随机调整图像的色度

    image = tf.image.random_hue(image, 0.02)



    # 随机调整图像的饱和度

    image = tf.image.random_saturation(image, 0.9, 1.2)



    # 随机左右翻转图像

    flip_left_right = lambda x: tf.image.random_flip_left_right(x)

    image = tf.map_fn(flip_left_right, image)



    # 随机上下翻转图像

    flip_up_down = lambda x: tf.image.random_flip_up_down(x)

    image = tf.map_fn(flip_up_down, image)



    # 将RGB格式转换为HSV格式

    hsv = tf.image.rgb_to_hsv(image)



    # 灰度图像合并到第四通道

    result = tf.concat([hsv, gray], axis=-1)

    return result
from keras.models import Model

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, Lambda, BatchNormalization

import tensorflow as tf





# 卷积神经网络模型

def fruit_model(image_shape, class_no):

    # 输入层

    img_input = Input(shape=image_shape, name='data')



    # 数据增强

    x = Lambda(augment_image)(img_input)



    # 第一卷积层

    x = Conv2D(16, (5, 5), strides=(1, 1), padding='same', name='conv1')(x)

    # 批量标准化

    x = BatchNormalization()(x)

    # 修正线性单元激励

    x = Activation('relu', name='conv1_relu')(x)

    # 池化

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool1')(x)



    # 第二卷积层

    x = Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv2')(x)

    # 批量标准化

    x = BatchNormalization()(x)

    # 修正线性单元激励

    x = Activation('relu', name='conv2_relu')(x)

    # 池化

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool2')(x)



    # 第三卷积层

    x = Conv2D(64, (5, 5), strides=(1, 1), padding='same', name='conv3')(x)

    # 批量标准化

    x = BatchNormalization()(x)

    # 修正线性单元激励

    x = Activation('relu', name='conv3_relu')(x)

    # 池化

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool3')(x)



    # 第四卷积层

    x = Conv2D(128, (5, 5), strides=(1, 1), padding='same', name='conv4')(x)

    # 批量标准化

    x = BatchNormalization()(x)

    # 修正线性单元激励

    x = Activation('relu', name='conv4_relu')(x)

    # 池化

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool4')(x)



    # 扁平化

    x = Flatten()(x)



    # 全连接层

    x = Dense(1024, activation='relu', name='fcl1')(x)

    x = Dropout(0.2)(x)



    # 全连接层

    x = Dense(128, activation='relu', name='fcl2')(x)

    x = Dropout(0.2)(x)



    # 输出层

    out = Dense(class_no, activation='softmax', name='predictions')(x)



    model = Model(inputs=img_input, outputs=out)



    print("卷积神经网络模型：")

    model.summary()

    return model
# 设置tf启用CPU的XLA功能，加快训练速度



import tensorflow as tf

config = tf.ConfigProto()

jit_level = tf.OptimizerOptions.ON_1

config.graph_options.optimizer_options.global_jit_level = jit_level

sess = tf.Session(config=config)

tf.keras.backend.set_session(sess)
import matplotlib.pyplot as plt

import keras

import numpy as np





class Graph(keras.callbacks.Callback):



    # This function is called when the training begins

    def on_train_begin(self, logs={}):

        # Initialize the lists for holding the logs, losses and accuracies

        self.losses = []

        self.acc = []

        self.val_losses = []

        self.val_acc = []

        self.logs = []



    # This function is called at the end of each epoch

    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists

        self.logs.append(logs)

        self.losses.append(logs.get('loss'))

        self.acc.append(logs.get('accuracy'))

        self.val_losses.append(logs.get('val_loss'))

        self.val_acc.append(logs.get('val_accuracy'))



        # Before plotting ensure at least 2 epochs have passed

        if len(self.losses) > 1:

            N = np.arange(1, len(self.losses) + 1)



            # You can chose the style of your preference

            # print(plt.style.available) to see the available options

            # plt.style.use("seaborn")



            # Plot train loss, train acc, val loss and val acc against epochs passed

            plt.figure()

            plt.plot(N, self.losses, label="train_loss")

            plt.plot(N, self.acc, label="train_acc")

            plt.plot(N, self.val_losses, label="val_loss")

            plt.plot(N, self.val_acc, label="val_acc")

            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch + 1))

            plt.xlabel("Epoch #")

            plt.ylabel("Loss/Accuracy")

            plt.legend()

            # Make sure there exists a folder called output in the current directory

            # or replace 'output' with whatever direcory you want to put in the plots

            # plt.savefig('output/Epoch-{}.png'.format(epoch))

            # plt.close()

            plt.show()
from keras.losses import sparse_categorical_crossentropy

import keras



# 卷积神经网络模型训练

def train():

    # 读取数据集

    (x_train, y_train), (x_validate, y_validate), class_dict, class_number = load_data('/kaggle/input/fruits360v14/fruits-360-v-14')



    # my_model实例化

    model = fruit_model(x_train[0].shape, class_number)



    # 优化训练参数

    adamax = keras.optimizers.Adamax(lr=0.0002, beta_1=0.9, beta_2=0.999, decay=0.0)

    

    # 编译模型

    model.compile(loss=sparse_categorical_crossentropy, optimizer=adamax, metrics=['accuracy'])



    # 绘图回调

    graph = Graph()

    

    # 训练模型

    model.fit(x_train, y_train, batch_size=128, epochs=30, shuffle=True, verbose=1, validation_data=(x_validate, y_validate), callbacks=[graph])



    # 保存模型

    model.save('model.h5')



    # 用测试集评估模型

    loss, accuracy = model.evaluate(x_validate, y_validate, batch_size=128)

    print()

    print("测试集损失值 = ",  loss)

    print("测试集正确率 = ",  accuracy)





# 运行训练

train()
