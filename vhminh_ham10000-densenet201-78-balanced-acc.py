import pandas as pd

import numpy as np

import os

import math

import itertools

from PIL import Image

from matplotlib import pyplot as plt

import keras

from keras.optimizers import Adam

from keras.layers import GlobalAveragePooling2D, Dense

from keras.models import Model

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from keras_applications.densenet import DenseNet201

from sklearn.model_selection import train_test_split

from sklearn.utils import compute_sample_weight, compute_class_weight

from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix



input_path = './../input'

metadata_path = os.path.join(input_path, 'HAM10000_metadata.csv')

part1_path = os.path.join(input_path, 'ham10000_images_part_1')

part2_path = os.path.join(input_path, 'ham10000_images_part_2')
label_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

data = pd.read_csv(metadata_path)

num_examples = data.values.shape[0]

inputs = np.empty(shape=(num_examples, 224, 224, 3), dtype=np.float32)

labels = np.empty(shape=(num_examples), dtype=np.uint8)



for i, row in enumerate(data.values):

    img_id = row[1]

    label = row[2]

    im_path = ''

    im_path1 = os.path.join(part1_path, img_id) + '.jpg'

    im_path2 = os.path.join(part2_path, img_id) + '.jpg'

    if (os.path.isfile(im_path1)):

        im_path = im_path1

    elif (os.path.isfile(im_path2)):

        im_path = im_path2

    else:

        raise Exception ('File not found \'%s\'' %img_id)

    img = Image.open(im_path).resize((224, 224), Image.LANCZOS)

    inputs[i] = np.array(img)/255

    labels[i] = label_names.index(label)

x_train, x_test, y_train, y_test = train_test_split(

    inputs, labels, test_size=0.2, random_state=2019)

del inputs

del labels

x_train, x_val, y_train, y_val = train_test_split(

    x_train, y_train, test_size=0.25, random_state=2019)
import keras.backend as K

from keras.legacy import interfaces

from keras.optimizers import Optimizer



class AdamAccumulate(Optimizer):



    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):

        if accum_iters < 1:

            raise ValueError('accum_iters must be >= 1')

        super(AdamAccumulate, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.lr = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay

        self.amsgrad = amsgrad

        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))

        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())



    @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]



        lr = self.lr



        completed_updates = K.cast(K.tf.floordiv(self.iterations, self.accum_iters), K.floatx())



        if self.initial_decay > 0:

            lr = lr * (1. / (1. + self.decay * completed_updates))



        t = completed_updates + 1



        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))



        # self.iterations incremented after processing a batch

        # batch:              1 2 3 4 5 6 7 8 9

        # self.iterations:    0 1 2 3 4 5 6 7 8

        # update_switch = 1:        x       x    (if accum_iters=4)  

        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)

        update_switch = K.cast(update_switch, K.floatx())



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]



        if self.amsgrad:

            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        else:

            vhats = [K.zeros(1) for _ in params]



        self.weights = [self.iterations] + ms + vs + vhats



        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):



            sum_grad = tg + g

            avg_grad = sum_grad / self.accum_iters_float



            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)



            if self.amsgrad:

                vhat_t = K.maximum(vhat, v_t)

                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)

                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))

            else:

                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)



            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))

            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))

            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))

        return self.updates



    def get_config(self):

        config = {'lr': float(K.get_value(self.lr)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'epsilon': self.epsilon,

                  'amsgrad': self.amsgrad}

        base_config = super(AdamAccumulate, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=(224,224,3),

                         backend=keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)



base_model.trainable = False



x = base_model.output

x = GlobalAveragePooling2D()(x)

preds = Dense(7,activation='softmax')(x)



model = Model(inputs=base_model.input,outputs=preds)



model.compile(optimizer=AdamAccumulate(lr=0.001, accum_iters=16), 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])





datagen = ImageDataGenerator(

        rotation_range=180,

        width_shift_range=0.1,

        height_shift_range=0.1,

        zoom_range=0.1,

        horizontal_flip=True,

        vertical_flip=True,

        fill_mode='nearest')



datagen.fit(x_train)



early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')

checkpoint_saving = ModelCheckpoint('model.hdf5', save_best_only=True, monitor='val_loss', mode='min')

reduce_lr_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.3, min_lr=0.00001 , patience=10, verbose=1, min_delta=1e-4, mode='min')

%%time



history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),

                              validation_data=(x_val, y_val, compute_sample_weight('balanced', y_val)),

                              class_weight=compute_class_weight('balanced', np.unique(y_train), y_train),

                              steps_per_epoch=math.ceil(x_train.shape[0]/16),

                              epochs=150,

                              callbacks=[early_stopping, checkpoint_saving, reduce_lr_rate])
model.load_weights('model.hdf5')

y_pred = np.argmax(model.predict(x_val), axis=1)

print('balanced acc on validation set:', balanced_accuracy_score(y_true=y_val, y_pred=y_pred))

y_pred = np.argmax(model.predict(x_test), axis=1)

print('balanced acc on test set:', balanced_accuracy_score(y_true=y_test, y_pred=y_pred))
def plot_confusion_matrix(matrix, labels,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    #print(matrix)



    plt.imshow(matrix, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(labels))

    plt.xticks(tick_marks, labels) #, rotation=45)

    plt.yticks(tick_marks, labels)



    fmt = '.2f' if normalize else 'd'

    thresh = matrix.max() / 2.

    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):

        plt.text(j, i, format(matrix[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if matrix[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()

    

    

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

plot_confusion_matrix(conf_matrix, labels=label_names, normalize=True)
report = classification_report(y_true=y_test, y_pred=y_pred, target_names=label_names)

print(report)