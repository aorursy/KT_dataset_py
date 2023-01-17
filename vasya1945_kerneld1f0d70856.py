import os
import numpy as np
import keras
from keras.models import Model
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

train_files = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] 
test_files = [TEST_DIR+i for i in os.listdir(TEST_DIR)] 
import re
from random import shuffle
from glob import glob

IMG_SIZE = (224, 224)  # размер входного изображения сети

# загружаем входное изображение и предобрабатываем
def load_image(path, target_size=IMG_SIZE):
    img = load_img(path, target_size=target_size)  # загрузка и масштабирование изображения
    array = img_to_array(img)
    return preprocess_input(array)  # предобработка для VGG16

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
%matplotlib inline
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(20, 20))
for i, path in enumerate(train_files[:10], 1):
    subplot = fig.add_subplot(i // 5 + 1, 5, i)
    plt.imshow(plt.imread(path));
    subplot.set_title('%s' % path.split('/')[-1]);
      
# base_model -  объект класса keras.models.Model (Functional Model)
base_model = MobileNet(include_top = False,
                   weights = 'imagenet',
                   input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))
# фиксируем все веса предобученной сети
for layer in base_model.layers:
    layer.trainable = False
base_model.summary()
reg = 1e-7

x = base_model.layers[-1].output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(reg))(x)
x = keras.layers.Dropout(rate=0.2)(x)
x = keras.layers.Dense(1,  # один выход
                activation='sigmoid',  # функция активации  
                kernel_regularizer=keras.regularizers.l1(1e-7))(x)
model = Model(inputs=base_model.input, outputs=x)
model.summary()
epochs = 30

def step_decay(epoch):
    lr = 0.000005
    step = 12
    return lr/10**int(epoch/step)

lrate = LearningRateScheduler(step_decay)
checkpointer = ModelCheckpoint(filepath="weights1.hdf5", verbose=1, save_best_only=True)

callbacks_list = [lrate, checkpointer]

opt = keras.optimizers.Adam()#, decay=decay_rate)
model.compile(optimizer=opt, 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
shuffle(train_files)  # перемешиваем обучающую выборку

train_val_split = 4000  # число изображений в валидационной выборке

validation_data = next(fit_generator(train_files[:train_val_split], train_val_split))

# запускаем процесс обучения
model.fit_generator(fit_generator(train_files[train_val_split:]),  # данные читаем функцией-генератором
        steps_per_epoch=150,  # число вызовов генератора за эпоху
        epochs=epochs,  # число эпох обучения
        validation_data=validation_data, callbacks=callbacks_list)
from sklearn.metrics import log_loss

#model.load_weights('weights1.hdf5')
#pr = model.predict(validation_data[0])
#print(log_loss(validation_data[1] , np.clip(pr, 0.00001, 0.9999)))

pred_ = model.predict_generator(predict_generator(test_files), len(test_files), max_queue_size=500)
pred = np.clip(pred_, 0.0000001, 0.9999999)
%matplotlib inline
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(20, 20))
for i, (path, score) in enumerate(zip(test_files[80:][15:25], pred[80:][15:25]), 1):
    subplot = fig.add_subplot(i // 5 + 1, 5, i)
    plt.imshow(plt.imread(path));
    subplot.set_title('%.3f' % score);
with open('submit.txt', 'w') as dst:
    dst.write('id,label\n')
    for path, score in zip(test_files, pred):
        dst.write('%s,%f\n' % (re.search('(\d+)', path).group(0), score))
