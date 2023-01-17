import numpy as np

%matplotlib inline

from matplotlib import pyplot as plt

from tensorflow import keras

from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_VGG16

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_ResNet50

from tensorflow.keras.preprocessing.image import load_img, img_to_array

import skimage

from random import randrange

import cv2

import re

from random import shuffle

from glob import glob

IMG_SIZE = (224, 224)  # размер входного изображения сети
class augmentator:

    def __init__(self):

        self.modes=["gaussian","localvar","poisson","salt","pepper","s&p","speckle",None]





    def rotate(self, image, angle=90, scale=1.0):

        w = image.shape[1]

        h = image.shape[0]

        #rotate matrix

        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)

        #rotate

        image = cv2.warpAffine(image,M,(w,h))

        return image



    def flip(self, image, vflip=False, hflip=False):

        if hflip or vflip:

            if hflip and vflip:

                c = -1

            else:

                c = 0 if vflip else 1

            image = cv2.flip(image, flipCode=c)

        return image 

    

    def add_noise(self, image, mode=0):

        if(self.modes[mode] == None):

            return image

        else:

            return skimage.util.random_noise(image, mode=self.modes[mode])

    

    def augment(self, image, typ): 

        img = image.copy()

        if(typ == 1):

            return self.flip(img, vflip=(randrange(2) == 1), hflip=(randrange(2) == 1))

        elif(typ == 2):

            return self.rotate(img, randrange(36)*10)

        elif(typ == 3):

            return self.add_noise(img, randrange(len(self.modes)))

        else:

            return img

aug = augmentator()
train_files = glob('../input/train/*.jpg')

test_files = glob('../input/test/*.jpg')



# загружаем входное изображение и предобрабатываем

def load_image(path, target_size=IMG_SIZE):

    img = load_img(path, target_size=target_size)  # загрузка и масштабирование изображения

    img = img_to_array(img)

    #img = cv2.imread(path)

    #img = cv2.resize(img, target_size)

    img = aug.augment(img, randrange(4)) # аугментация

    #return preprocess_input_VGG16(img)  # предобработка для VGG16

    return preprocess_input_ResNet50(img)  # предобработка для ResNet50



# генератор для последовательного чтения обучающих данных с диска

def fit_generator(files, batch_size=32):

    while True:

        shuffle(files)

        for k in range(len(files) // batch_size):

            i = k * batch_size

            j = i + batch_size

            if j > len(files):

                j = - j % len(files)

            x = np.array([load_image(path) for path in files[i:j]])

            y = np.array([1. if re.match('.*/dog\.\d', path) else 0. for path in files[i:j]])

            yield (x, y)



# генератор последовательного чтения тестовых данных с диска

def predict_generator(files):

    while True:

        for path in files:

            yield np.array([load_image(path)])
fig = plt.figure(figsize=(20, 20))

for i, path in enumerate(train_files[:10], 1):

    subplot = fig.add_subplot(i // 5 + 1, 5, i)

    plt.imshow(plt.imread(path));

    subplot.set_title('%s' % path.split('/')[-1]);
fig = plt.figure(figsize=(20, 20))

for i, path in enumerate(train_files[:10], 1):

    subplot = fig.add_subplot(i // 5 + 1, 5, i)

    img = cv2.imread(path)

    img = cv2.resize(img, IMG_SIZE)

    img = aug.augment(img, randrange(4))

    plt.imshow(img);

    subplot.set_title('%s' % path.split('/')[-1]);
# base_model -  объект класса keras.models.Model (Functional Model)

#base_model = VGG16(include_top = False,

#                   weights = 'imagenet',

#                   input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))



base_model = ResNet50(include_top = False,

                   weights = 'imagenet',

                   input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))

# фиксируем все веса предобученной сети

for layer in base_model.layers:

    layer.trainable = False
base_model.summary()
x = base_model.layers[-5].output

x = keras.layers.Flatten()(x)

x = keras.layers.Dense(1,  # один выход

                activation='sigmoid',  # функция активации  

                kernel_regularizer=keras.regularizers.l1(1e-4))(x)

model = Model(inputs=base_model.input, outputs=x)
model.summary()
model.compile(optimizer='adam', 

              loss='binary_crossentropy',  # функция потерь binary_crossentropy (log loss

              metrics=['accuracy'])
shuffle(train_files)  # перемешиваем обучающую выборку



train_val_split = 100  # число изображений в валидационной выборке



validation_data = next(fit_generator(train_files[:train_val_split], train_val_split))



# запускаем процесс обучения

model.fit_generator(fit_generator(train_files[train_val_split:]),  # данные читаем функцией-генератором

        steps_per_epoch=10,  # число вызовов генератора за эпоху

        epochs=100,  # число эпох обучения

        validation_data=validation_data)
#model.save('cats-dogs-vgg16.hdf5')

model.save('cats-dogs-resnet05.hdf5')
pred = model.predict_generator(predict_generator(test_files), len(test_files), max_queue_size=500)
fig = plt.figure(figsize=(20, 20))

for i, (path, score) in enumerate(zip(test_files[80:][:10], pred[80:][:10]), 1):

    subplot = fig.add_subplot(i // 5 + 1, 5, i)

    plt.imshow(plt.imread(path));

    subplot.set_title('%.3f' % score);
with open('submit-0717.csv', 'w') as dst:

    dst.write('id,label\n')

    for path, score in zip(test_files, pred):

        dst.write('%s,%f\n' % (re.search('(\d+)', path).group(0), score))
# LogLoss = 0.013102