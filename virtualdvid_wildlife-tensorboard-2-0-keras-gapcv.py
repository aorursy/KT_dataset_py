%%capture

# install tensorflow 2.0 alpha

!pip install -q tensorflow-gpu==2.0.0-alpha0



#install GapCV

!pip install -q gapcv
import os

import time

import cv2

import gc

import numpy as np



import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras import layers

from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras import regularizers

from tensorflow.keras import callbacks



import gapcv

from gapcv.vision import Images



from sklearn.utils import class_weight



import warnings

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



print('tensorflow version: ', tf.__version__)

print('keras version: ', tf.keras.__version__)

print('gapcv version: ', gapcv.__version__)



from matplotlib import pyplot as plt

%matplotlib inline



os.makedirs('model', exist_ok=True)

print(os.listdir('../input'))

print(os.listdir('./'))
def elapsed(start):

    """

    Returns elapsed time in hh:mm:ss format from start time in unix format

    """

    elapsed = time.time()-start

    return time.strftime("%H:%M:%S", time.gmtime(elapsed))
def plot_sample(imgs_set, labels_set, img_size=(12,12), columns=4, rows=4, random=False):

    """

    Plot a sample of images

    """

    

    fig=plt.figure(figsize=img_size)

    

    for i in range(1, columns*rows + 1):

        

        if random:

            img_x = np.random.randint(0, len(imgs_set))

        else:

            img_x = i-1

        

        img = imgs_set[img_x]

        ax = fig.add_subplot(rows, columns, i)

        ax.set_title(str(labels_set[img_x]))

        plt.axis('off')

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.show()
data_set = 'wildlife'

data_set_folder = 'oregon_wildlife/oregon_wildlife'

minibatch_size = 32



if not os.path.isfile('../input/{}.h5'.format(data_set)):

    images = Images(data_set, data_set_folder, config=['resize=(128,128)', 'store', 'stream'])



# stream from h5 file

images = Images(config=['stream'], augment=['flip=horizontal', 'edge', 'zoom=0.3', 'denoise'])

images.load(data_set, '../input')



# generator

images.split = 0.2

X_test, Y_test = images.test

images.minibatch = minibatch_size

gap_generator = images.minibatch



Y_int = [y.argmax() for y in Y_test]

class_weights = class_weight.compute_class_weight(

    'balanced',

    np.unique(Y_int),

    Y_int

)



total_train_images = images.count - len(X_test)

n_classes = len(images.classes)
print('content:', os.listdir("./"))

print('time to load data set:', images.elapsed)

print('number of images in data set:', images.count)

print('classes:', images.classes)

print('data type:', images.dtype)
!free -m
model = Sequential([

    layers.Conv2D(filters=128, kernel_size=(4, 4), activation='tanh', input_shape=(128, 128, 3)),

    layers.MaxPool2D(pool_size=(2,2)),

    layers.Dropout(0.22018745727040784),

    layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu'),

    layers.MaxPool2D(pool_size=(2,2)),

    layers.Dropout(0.02990527559235584),

    layers.Conv2D(filters=32, kernel_size=(4, 4), activation='tanh'),

    layers.MaxPool2D(pool_size=(2,2)),

    layers.Dropout(0.0015225556862044631),

    layers.Conv2D(filters=32, kernel_size=(4, 4), activation='tanh'),

    layers.MaxPool2D(pool_size=(2,2)),

    layers.Dropout(0.1207251417283281),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),

    layers.Dropout(0.4724418446300173),

    layers.Dense(5, activation='softmax')

])
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_file = './model/model.h5'



# # Clear any logs from previous runs

# !rm -rf ./logs/fit/*

# !rm -rf ./model/*



# log_dir="./logs/fit/{}".format(time.strftime("%Y%m%d-%H%M%S", time.gmtime()))



# get_ipython().system_raw(

#     'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'

#     .format(log_dir)

# )
## commented out if you want to check the localhost

# !curl http://localhost:6006
# !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip > /dev/null 2>&1

# !unzip ngrok-stable-linux-amd64.zip > /dev/null 2>&1

    

# get_ipython().system_raw('./ngrok http 6006 &')
# !curl -s http://localhost:4040/api/tunnels | python3 -c \

#     "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
# tensorboard = callbacks.TensorBoard(

#     log_dir=log_dir,

#     histogram_freq=1

# )



earlystopping = callbacks.EarlyStopping(

    monitor='val_loss',

    patience=5

)



model_checkpoint = callbacks.ModelCheckpoint(

    model_file,

    monitor='val_accuracy',

    save_best_only=True,

    save_weights_only=False,

    mode='max'

)
start = time.time()



model.fit_generator(

    generator=gap_generator,

    validation_data=(X_test, Y_test),

    epochs=50,

    steps_per_epoch=int(total_train_images / minibatch_size),

    initial_epoch=0,

    verbose=1,

    class_weight=class_weights,

    callbacks=[

        # tensorboard,

        model_checkpoint

    ]

)



print('\nElapsed time: {}'.format(elapsed(start)))
del model

model = load_model(model_file)
scores = model.evaluate(X_test, Y_test, batch_size=32)



for score, metric_name in zip(scores, model.metrics_names):

    print("{} : {}".format(metric_name, score))
!curl https://d36tnp772eyphs.cloudfront.net/blogs/1/2016/11/17268317326_2c1525b418_k.jpg > test_image.jpg
labels = {val:key for key, val in images.classes.items()}

labels
%pwd

%ls
image2 = Images('foo', ['test_image.jpg'], [0], config=['resize=(128,128)'])

img = image2._data[0]
prediction = model.predict_classes(img)

prediction = labels[prediction[0]]



plot_sample(img, ['predicted image: {}'.format(prediction)], img_size=(8, 8), columns=1, rows=1)