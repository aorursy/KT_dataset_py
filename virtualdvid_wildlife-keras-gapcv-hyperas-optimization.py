%%capture

# install tensorflow 2.0 beta

!pip install -q tensorflow-gpu==2.0.0-beta0



# install GapCV

!pip install -q gapcv



# install hyperas

!pip install -q hyperas
# import libraries

import os

from hyperas import optim

from hyperopt import Trials, STATUS_OK, tpe

from hyperas.distributions import choice, uniform



import warnings

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



print(os.listdir('../input'))

print(os.listdir('./'))
# Meta data

def metadata():

    return {

        'data_set': 'wildlife',

        'data_set_folder': 'oregon_wildlife/oregon_wildlife',

        'img_height': 128,

        'img_width': 128,

        'test_batch_size': 2,

        'nb_epochs': 100

    }
def gap():

    import os

    from gapcv.vision import Images



    info=metadata()

    data_set = info['data_set']

    data_set_folder = info['data_set_folder']

    img_height = info['img_height']

    img_width = info['img_width']



    if not os.path.isfile('../input/{}.h5'.format(data_set)):

        print('image preprocessing started...')

        images = Images(

            '../input/{}'.format(data_set_folder),

            config=[

                'resize=({},{})'.format(img_height,img_width),

                'stream',

            ]

        )

        print('Time elapsed to preprocess the data set:', images.elapsed)



    # load train_wildlife.h5 if exist

    images = Images(

        config=['stream'],

        augment=[

            'flip=horizontal',

            'edge',

            'zoom=0.3',

            'denoise'

        ]

    )

    images.load(data_set, '../input')

    return images
def gap_generator(minibatch, images):

    images.minibatch = minibatch

    gap_generator = images.minibatch

    return gap_generator
def data():

    import numpy as np

    from sklearn.utils import class_weight



    images = gap()



    images.split = 0.2

    X_test, Y_test = images.test



    Y_int = [y.argmax() for y in Y_test]

    class_weights = class_weight.compute_class_weight(

        'balanced',

        np.unique(Y_int),

        Y_int

    )

    generator = gap_generator



    return generator, X_test, Y_test, class_weights, images
def model_op(generator, X_test, Y_test, class_weights, images):



    # import libraries

    import os

    import gc

    import secrets



    import tensorflow as tf

    from tensorflow.keras import backend as K

    from tensorflow.keras.models import Sequential, load_model

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    from tensorflow.keras import layers, regularizers, callbacks

    from tensorflow.keras.optimizers import SGD, Adam



    class StopTraining(callbacks.Callback):

        def __init__(self, monitor='val_loss', patience=10, goal=0.5):

            self.monitor = monitor

            self.patience = patience

            self.goal = goal



        def on_epoch_end(self, epoch, logs={}):

            current_val_acc = logs.get(self.monitor)



            if current_val_acc < self.goal and epoch == self.patience:

                self.model.stop_training = True

      

    info=metadata()

    nb_epochs = info['nb_epochs']

    img_height = info['img_height']

    img_width = info['img_width']

    

    n_classes = len(images.classes)

    total_train_images = images.count - len(X_test)



    os.makedirs('model', exist_ok=True)

    model_name = secrets.token_hex(5)



    try:

        model = Sequential()



        # Section 1

        model.add(

            layers.Conv2D(

                filters={{choice([32, 64, 128, 256, 512])}},

                kernel_size={{choice([3, 4])}},

                activation='relu',

                input_shape=(img_height,img_width,3)

            )

        )

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Dropout({{uniform(0, .9)}}))



        # Section 2

        model.add(

            layers.Conv2D(

                filters={{choice([32, 64, 128, 256, 512])}},

                kernel_size={{choice([3, 4])}},

                activation='relu'

            )

        )

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Dropout({{uniform(0, .9)}}))

        

        # Section 3

        model.add(layers.Flatten())

        model.add(layers.Dense({{choice([32, 64, 128, 256, 512])}}, activation='relu'))

        model.add(layers.Dropout({{uniform(0, .9)}}))

        model.add(layers.Dense(n_classes, activation='softmax'))



        model.compile(

            loss='categorical_crossentropy',

            metrics=['accuracy'],

            optimizer='sgd'

        )



        earlystopping = callbacks.EarlyStopping(

            monitor='val_loss',

            patience=5

        )

        model_file = 'model/{}.h5'.format(model_name)

        model_checkpoint = callbacks.ModelCheckpoint(

            model_file,

            monitor='val_accuracy',

            save_best_only=True,

            save_weights_only=False, mode='max'

        )

        stoptraining = StopTraining(

            monitor='val_accuracy',

            patience=30,

            goal=0.6

        )



        minibatch_size = {{choice([16, 32, 64, 128])}}

        print('trainings started...')

        model.fit_generator(

            generator=generator(minibatch_size, images),

            validation_data=(X_test, Y_test),

            epochs=nb_epochs,

            steps_per_epoch=int(total_train_images / minibatch_size),

            initial_epoch=0,

            verbose=0,

            class_weight=class_weights,

            callbacks=[

                model_checkpoint,

                earlystopping,

                stoptraining

            ]

        )

        

        model = load_model(model_file)



        score, acc = model.evaluate(X_test, Y_test, verbose=0)

        print('Test accuracy:', acc)



        # delete model file

        os.remove(model_file)

    except Exception as e:

        acc = 0.0

        print('failed', e)

    

    del model

    K.clear_session()

    for _ in range(12):

        gc.collect()

    

    return {'loss': -acc, 'status': STATUS_OK}
best_run, best_model = optim.minimize(

    model=model_op,

    data=data,

    functions=[gap, gap_generator, metadata],

    algo=tpe.suggest,

    max_evals=20,

    trials=Trials(),

    notebook_name='__notebook__' # __notebook_source__ or __notebook__

)
best_run