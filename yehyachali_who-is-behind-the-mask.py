import numpy as np
import matplotlib.pyplot as plt
from keras.utils.data_utils import Sequence
import keras
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, Flatten, Reshape, Dense, add
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import cv2
import random, math
masks = [f"../input/face-mask-lite-dataset/with_mask/with-mask-default-mask-seed{str(i).rjust(4,'0')}.png" for i in range(1000)]
nomasks = [f"../input/face-mask-lite-dataset/without_mask/seed{str(i).rjust(4,'0')}.png" for i in range(1000)]
class Daugmentation:
    def flip(img, t):
        if t[0]==0:
            return img
        else:
            return cv2.flip(img, t[1])

    def zoom(img, t):
        if t[2]==0:
            return img
        else:
            h, w = img.shape[:2]
            nh, nw =  int(t[3]*h), int(t[3]*w)
            dh, dw = h-nh, w-nw
            zimg = img[dh//2:nh+dh//2, dw//2:nw+dw//2]
            zimg = cv2.resize(zimg, (w,h))
            return zimg


    def get_ts(batch_size):
        return [[random.choice([0,1]),random.choice([-1,0,1]), random.randint(0,2),random.uniform(0.4,0.9)] for i in range(batch_size)]

    def aug(img,t):
        img = Daugmentation.flip(img,t)
        # img = Daugmentation.zoom(img, t)
        return img


class maSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, Daugmentation):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = Daugmentation

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        ts = self.augment.get_ts(len(batch_x))
        return np.array([self.augment.aug(x,t) for x,t in zip(batch_x,ts)]),np.array([self.augment.aug(y,t) for y,t in zip(batch_y,ts)]) 
X = []
Y = []
test = []
for i in zip(masks, nomasks):
    img = load_img(i[0], target_size=(128,128))
    X.append(img_to_array(img)/250.)
    img = load_img(i[1], target_size=(128,128))
    Y.append(img_to_array(img)/250.)

X = np.array(X)
Y = np.array(Y)
np.random.seed(777)
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1)
x_train.shape,y_train.shape, x_test.shape, y_test.shape
plt.figure(figsize=(20,5))
masks = x_test[:10]
nomasks = y_test[:10]
for i, (x,y) in enumerate(zip(masks[:10],nomasks[:10])):
    plt.subplot(2,10,i+1)
    plt.imshow(x)
    plt.axis("OFF")
    
    plt.subplot(2,10,i+11)
    plt.imshow(y)
    plt.axis("OFF")
plt.show()
batch_size=32

gotrain = maSequence(x_train, y_train, batch_size,Daugmentation)
gotest = maSequence(x_test, y_test, batch_size,Daugmentation)

def get_model():
    # encoder
    In = Input(shape=x_train[0].shape)
    c1 = Conv2D(32, 3, activation="relu", padding="same")(In)
    c1 = Conv2D(32, 3, activation="relu", padding="same")(c1)
#     c1 = Conv2D(32, 3, activation="relu", padding="same")(c1)
#     c1 = Conv2D(32, 3, activation="relu", padding="same")(c1)
    m1 = MaxPooling2D(2)(c1)
    c2 = Conv2D(64, 3, activation="relu", padding="same")(m1)
    c2 = Conv2D(64, 3, activation="relu", padding="same")(c2)
#     c2 = Conv2D(64, 3, activation="relu", padding="same")(c2)
#     c2 = Conv2D(64, 3, activation="relu", padding="same")(c2)
    m2 = MaxPooling2D(2)(c2)
    c3 = Conv2D(128, 3, activation="relu", padding="same")(m2)
    c3 = Conv2D(128, 3, activation="relu", padding="same")(c3)
#     c3 = Conv2D(128, 3, activation="relu", padding="same")(c3)
#     c3 = Conv2D(128, 3, activation="relu", padding="same")(c3)
    m3 = MaxPooling2D(2)(c3)
    c4 = Conv2D(256, 3, activation="relu", padding="same")(m3)
    c4 = Conv2D(256, 3, activation="relu", padding="same")(c4)
#     c4 = Conv2D(256, 3, activation="relu", padding="same")(c4)
#     c4 = Conv2D(256, 3, activation="relu", padding="same")(c4)
    u1 = Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(c4)
    c5 = Conv2D(128, 3, activation="relu", padding="same")(u1)
    c5 = Conv2D(128, 3, activation="relu", padding="same")(c5)
#     c5 = Conv2D(128, 3, activation="relu", padding="same")(c5)
#     c5 = Conv2D(128, 3, activation="relu", padding="same")(c5)
    a1 = add([c5,c3])
    u2 = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(a1)
    c6 = Conv2D(64, 3, activation="relu", padding="same")(u2)
    c6 = Conv2D(64, 3, activation="relu", padding="same")(c6)
#     c6 = Conv2D(64, 3, activation="relu", padding="same")(c6)
#     c6 = Conv2D(64, 3, activation="relu", padding="same")(c6)
    a2 = add([c6,c2])
    u3 = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(a2)
    c7 = Conv2D(32, 3, activation="relu", padding="same")(u3)
    c7 = Conv2D(32, 3, activation="relu", padding="same")(c7)
#     c7 = Conv2D(32, 3, activation="relu", padding="same")(c7)
#     c7 = Conv2D(32, 3, activation="relu", padding="same")(c7)
    a3 = add([c7,c1])
    Out = Conv2D(3, 3, activation="sigmoid", padding="same")(a3)
    
    model = Model(In,Out)
    # adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer="adam",loss="binary_crossentropy")
    return model

model = get_model()
model.summary()
checkpointer = ModelCheckpoint(filepath='best.h5', verbose=1, save_best_only=True)
history = model.fit_generator(generator=gotrain,
                              steps_per_epoch=gotrain.__len__(),
                              epochs=100,
                              verbose=0,
                              callbacks=[checkpointer],
                              validation_data=gotest,
                              validation_steps=gotest.__len__())
plt.figure(figsize=(20,20))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss","val_loss"])
plt.show()
plt.figure(figsize=(20,5))
masks = x_test[:10]
nomasks = model.predict(masks)
for i, (x,y) in enumerate(zip(masks[:10],nomasks[:10])):
    plt.subplot(2,10,i+1)
    plt.imshow(x)
    plt.axis("OFF")
    
    plt.subplot(2,10,i+11)
    plt.imshow(y)
    plt.axis("OFF")
plt.show()



