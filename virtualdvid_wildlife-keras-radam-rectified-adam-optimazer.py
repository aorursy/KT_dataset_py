%%capture

# install tensorflow 2.0 beta

!pip install -q tensorflow-gpu==2.0.0-beta1

# !pip install -q tensorflow-gpu==2.0.0-rc0 # issue https://github.com/tensorflow/tensorflow/issues/24828



#install GapCV

!pip install -q gapcv
%%capture

import os

import time

import cv2

import gc

import numpy as np



import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras import layers

from tensorflow.keras.optimizers import SGD, Adam, Optimizer

from tensorflow.keras import regularizers

from tensorflow.keras import callbacks



import gapcv

from gapcv.vision import Images



from sklearn.utils import class_weight



import warnings

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['TF_KERAS']='1'



from matplotlib import pyplot as plt

%matplotlib inline
print('tensorflow version: ', tf.__version__)

print('keras version: ', tf.keras.__version__)

print('gapcv version: ', gapcv.__version__)



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
def plot_history(history, val_1, val_2, title):

    plt.plot(history.history[val_1])

    plt.plot(history.history[val_2])



    plt.title(title)

    plt.ylabel(val_1)

    plt.xlabel('epoch')

    plt.legend([val_1, val_2], loc='upper left')

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
# dataset meta data

print('content:', os.listdir("./"))

print('time to load data set:', images.elapsed)

print('number of images in data set:', images.count)

print('classes:', images.classes)

print('data type:', images.dtype)
!free -m
model_file = './model/model.h5'



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
def model_seq():

    return Sequential([

        layers.Conv2D(filters=128, kernel_size=(4, 4), activation='relu', input_shape=(128, 128, 3)),

        layers.MaxPool2D(pool_size=(2,2)),

        layers.Dropout(0.3),

        layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu'),

        layers.MaxPool2D(pool_size=(2,2)),

        layers.Dropout(0.1),

        layers.Conv2D(filters=32, kernel_size=(4, 4), activation='relu'),

        layers.MaxPool2D(pool_size=(2,2)),

        layers.Dropout(0.1),

        layers.Conv2D(filters=32, kernel_size=(4, 4), activation='relu'),

        layers.MaxPool2D(pool_size=(2,2)),

        layers.Dropout(0.2),

        layers.Flatten(),

        layers.Dense(256, activation='relu'),

        layers.Dropout(0.5),

        layers.Dense(5, activation='softmax')

    ])
model = model_seq()

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
start = time.time()



history = model.fit_generator(

    generator=gap_generator,

    validation_data=(X_test, Y_test),

    epochs=100,

    steps_per_epoch=int(total_train_images / minibatch_size),

    initial_epoch=0,

    verbose=1,

    class_weight=class_weights,

    callbacks=[

        model_checkpoint

    ]

)
# moved line for kernel bug

print('\nElapsed time: {}'.format(elapsed(start)))
plot_history(history, 'accuracy', 'val_accuracy', 'Accuracy')

plot_history(history, 'loss', 'val_loss', 'Loss')
del model

model = load_model(model_file)
scores = model.evaluate(X_test, Y_test, batch_size=32)



for score, metric_name in zip(scores, model.metrics_names):

    print("{} : {}".format(metric_name, score))
class RAdam(Optimizer):

    """RAdam optimizer.

    # Arguments

        lr: float >= 0. Learning rate.

        beta_1: float, 0 < beta < 1. Generally close to 1.

        beta_2: float, 0 < beta < 1. Generally close to 1.

        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.

        decay: float >= 0. Learning rate decay over each update.

        weight_decay: float >= 0. Weight decay for each param.

    # References

        - [Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)

        - [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf)

    """



    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0.0, weight_decay=0.0, **kwargs):

        super(RAdam, self).__init__(name='RAdam', **kwargs)

        with K.name_scope(self.__class__.__name__):

            self._lr = K.variable(lr, name='lr')

            self._iterations = K.variable(0, dtype='int64', name='iterations')

            self._beta_1 = K.variable(beta_1, name='beta_1')

            self._beta_2 = K.variable(beta_2, name='beta_2')

            self._decay = K.variable(decay, name='decay')

            self._weight_decay = K.variable(weight_decay, name='weight_decay')

        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay

        self.initial_weight_decay = weight_decay



    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self._iterations, 1)]

        lr = self._lr

        if self.initial_decay > 0:

            lr = lr * (1. / (1. + self._decay * K.cast(self._iterations, K.dtype(self._decay))))



        t = K.cast(self._iterations, K.floatx()) + 1



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i)) for (i, p) in enumerate(params)]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i)) for (i, p) in enumerate(params)]



        self._weights = [self._iterations] + ms + vs



        beta_1_t = K.pow(self._beta_1, t)

        beta_2_t = K.pow(self._beta_2, t)



        sma_inf = 2.0 / (1.0 - self._beta_2) - 1.0

        sma_t = sma_inf - 2.0 * t * beta_2_t / (1.0 - beta_2_t)



        for p, g, m, v in zip(params, grads, ms, vs):

            m_t = (self._beta_1 * m) + (1. - self._beta_1) * g

            v_t = (self._beta_2 * v) + (1. - self._beta_2) * K.square(g)



            m_hat_t = m_t / (1.0 - beta_1_t)

            v_hat_t = K.sqrt(v_t / (1.0 - beta_2_t) + self.epsilon)



            r_t = K.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *

                         (sma_t - 2.0) / (sma_inf - 2.0) *

                         sma_inf / sma_t + self.epsilon)



            p_t = K.switch(sma_t > 5, r_t * m_hat_t / (K.sqrt(v_hat_t + self.epsilon)), m_hat_t)



            if self.initial_weight_decay > 0:

                p_t += self._weight_decay * p



            p_t = p - lr * p_t



            self.updates.append(K.update(m, m_t))

            self.updates.append(K.update(v, v_t))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, new_p))

        return self.updates



    def get_config(self):

        config = {

            'lr': float(K.get_value(self._lr)),

            'beta_1': float(K.get_value(self._beta_1)),

            'beta_2': float(K.get_value(self._beta_2)),

            'decay': float(K.get_value(self._decay)),

            'weight_decay': float(K.get_value(self._weight_decay)),

            'epsilon': self.epsilon,

        }

        base_config = super(RAdam, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
del model

model = model_seq()
model.compile(optimizer=RAdam(), loss='categorical_crossentropy', metrics=['accuracy'])
start = time.time()



history = model.fit_generator(

    generator=gap_generator,

    validation_data=(X_test, Y_test),

    epochs=100,

    steps_per_epoch=int(total_train_images / minibatch_size),

    initial_epoch=0,

    verbose=1,

    class_weight=class_weights,

    callbacks=[

        model_checkpoint

    ]

)
# moved line for kernel bug

print('\nElapsed time: {}'.format(elapsed(start)))
plot_history(history, 'accuracy', 'val_accuracy', 'Accuracy')

plot_history(history, 'loss', 'val_loss', 'Loss')
del model

model = load_model(model_file)
scores = model.evaluate(X_test, Y_test, batch_size=32)



for score, metric_name in zip(scores, model.metrics_names):

    print("{} : {}".format(metric_name, score))
del model

model = model_seq()
model.compile(optimizer=RAdam(), loss='categorical_crossentropy', metrics=['accuracy'])
start = time.time()



history = model.fit_generator(

    generator=gap_generator,

    validation_data=(X_test, Y_test),

    epochs=200,

    steps_per_epoch=int(total_train_images / minibatch_size),

    initial_epoch=0,

    verbose=1,

    class_weight=class_weights,

    callbacks=[

        model_checkpoint

    ]

)
# moved line for kernel bug

print('\nElapsed time: {}'.format(elapsed(start)))
plot_history(history, 'accuracy', 'val_accuracy', 'Accuracy')

plot_history(history, 'loss', 'val_loss', 'Loss')
# handle unexpected bug - ValueError: Unknown optimizer: RAdam

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