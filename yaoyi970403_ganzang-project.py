import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
#图片大小
im_width = 416
im_height = 416

BATCH_SIZE = 8

train_files = '../input/ganzang/data/train/'
val_files = '../input/ganzang/data/val/'

if not os.path.exists("save_weights"):
    os.makedirs("save_weights")


train_num = len(os.listdir(os.path.join(train_files,'image')))
val_num = len(os.listdir(os.path.join(val_files,'image')))
# 定义数据生成器函数
def data_generator(data_path, batch_size, aug_dict,
                   image_color_mode="rgb",
                   mask_color_mode="grayscale",
                   image_save_prefix="image",
                   mask_save_prefix="mask",
                   save_to_dir=None,
                   target_size=(im_width, im_height),
                   seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        data_path,
        classes=['image'],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        data_path,
        classes=['label'],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_gen = zip(image_generator, mask_generator)

    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img, mask)


# 处理数据函数
def adjust_data(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    return (img, mask)


smooth = 100


# 定义Dice系数
def dice_coef(y_true, y_pred):
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)
    return ((2 * And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))


# 定义损失函数
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# 定义iou函数
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


def jac_distance(y_true, y_pred):
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)

    return - iou(y_true, y_pred)


#定义U-Net网络模型
def unet(input_size=(416, 416, 3)):
    inputs = Input(input_size)

    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    bn1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same')(bn1)
    bn1 = BatchNormalization(axis=3)(conv1)
    bn1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    bn2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same')(bn2)
    bn2 = BatchNormalization(axis=3)(conv2)
    bn2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    bn3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), padding='same')(bn3)
    bn3 = BatchNormalization(axis=3)(conv3)
    bn3 = Activation('relu')(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    bn4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3, 3), padding='same')(bn4)
    bn4 = BatchNormalization(axis=3)(conv4)
    bn4 = Activation('relu')(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
    bn5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, (3, 3), padding='same')(bn5)
    bn5 = BatchNormalization(axis=3)(conv5)
    bn5 = Activation('relu')(bn5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), padding='same')(up6)
    bn6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, (3, 3), padding='same')(bn6)
    bn6 = BatchNormalization(axis=3)(conv6)
    bn6 = Activation('relu')(bn6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), padding='same')(up7)
    bn7 = Activation('relu')(conv7)
    conv7 = Conv2D(256, (3, 3), padding='same')(bn7)
    bn7 = BatchNormalization(axis=3)(conv7)
    bn7 = Activation('relu')(bn7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), padding='same')(up8)
    bn8 = Activation('relu')(conv8)
    conv8 = Conv2D(128, (3, 3), padding='same')(bn8)
    bn8 = BatchNormalization(axis=3)(conv8)
    bn8 = Activation('relu')(bn8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), padding='same')(up9)
    bn9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, (3, 3), padding='same')(bn9)
    bn9 = BatchNormalization(axis=3)(conv9)
    bn9 = Activation('relu')(bn9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(bn9)

    return Model(inputs=[inputs], outputs=[conv10])
model = unet()
# 打印模型参数
model.summary()

#训练集图片做数据增强
train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')
#生成训练数据
train_gen = data_generator(train_files, BATCH_SIZE,
                                train_generator_args,
                                target_size=(im_height, im_width))
#生成验证数据
val_gen = data_generator(val_files, BATCH_SIZE,
                                dict(),
                                target_size=(im_height, im_width))

checkpoint = ModelCheckpoint(
                                filepath='./save_weights/myUnet.h5',
                                monitor='val_acc',
                                save_weights_only=False,
                                save_best_only=True,
                                mode='auto',
                                period=1
                            )

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)

#编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=dice_coef_loss, metrics=["binary_accuracy", iou, dice_coef])
history = model.fit(train_gen,
                    steps_per_epoch= train_num // BATCH_SIZE,
                    epochs=30,
                    validation_data = val_gen,
                    validation_steps= val_num // BATCH_SIZE,
                    callbacks=[checkpoint,reduce_lr])
model.save_weights('./save_weights/myUnet.h5')
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
val_img_files = '../input/ganzang/data/val/image/'
val_label_files = '../input/ganzang/data/val/label/'

images = os.listdir(val_img_files)
mask = os.listdir(val_label_files)
model = unet()
model.load_weights('./save_weights/myUnet.h5')
prediction_overlap = []
for i in range(20):
    img = cv2.imread(os.path.join(val_img_files,images[i]))
    img = cv2.resize(img ,(416, 416))
    img1 = img / 255
    img1 = img1[np.newaxis, :, :, :]
    prediction=model.predict(img1)
    prediction = np.squeeze(prediction)
    ground_truth = cv2.resize(cv2.imread(os.path.join(val_label_files,mask[i])),(416,416))
    ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
    ret, thresh_gt= cv2.threshold(ground_truth, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    overlap_img = cv2.drawContours(img, contours, -1, (0, 255, 0),2)

    prediction[np.nonzero(prediction < 0.3)] = 0.0
    prediction[np.nonzero(prediction >= 0.3)] = 255.
    prediction = prediction.astype("uint8")
    ret_p, thresh_p = cv2.threshold(prediction, 127, 255, 0)
    contours_p, hierarchy_p = cv2.findContours(thresh_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    overlap_img = cv2.drawContours(img, contours_p, -1, (255,36,0),2)
    prediction_overlap.append(overlap_img)

plt.figure(figsize=(20,20))
for i in range(len(prediction_overlap)):
    plt.subplot(4,5,i+1)
    plt.imshow(prediction_overlap[i])
    plt.title('Predict Image')
    plt.xticks([])
    plt.yticks([])

plt.show()
