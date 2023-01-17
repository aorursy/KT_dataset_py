from itertools import product

from math import ceil

from string import ascii_uppercase

from time import time



import numpy as np

import pandas as pd



from numpy.random import seed

seed(27)

from tensorflow import set_random_seed

set_random_seed(27)



from keras import backend as K

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization

from keras.models import Sequential, load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical



from sklearn.metrics import classification_report, confusion_matrix



from matplotlib import pyplot as plt

import seaborn as sns

sns.set()
def get_csv_data(path):

    dataframe = pd.read_csv(path)



    labels = dataframe['label'].values

    labels_categorical = to_categorical(labels)

    dataframe.drop('label', axis=1, inplace = True)

    

    images = dataframe.values

    images = images / 255

    images = np.array([np.reshape(i, (28, 28, 1)) for i in images])



    return images, labels, labels_categorical
train_images, train_labels, train_labels_categorical = get_csv_data('../input/sign_mnist_train.csv')

test_images, test_labels, test_labels_categorical = get_csv_data('../input/sign_mnist_test.csv')
test = pd.read_csv('../input/sign_mnist_test.csv')

labels_indices = np.unique(np.array(test['label'].values))

classes = [ascii_uppercase[i] for i in labels_indices]
def get_augmented_data_flow(images, labels_categorical, batch_size, validation_split=0.0, subset=None):

#     generator = ImageDataGenerator(

#         rotation_range=15,

#         width_shift_range=0.1,

#         height_shift_range=0.1,

#         shear_range=.05,

#         zoom_range=.05,

#         brightness_range=[0.1,1.0],

#         horizontal_flip=True,

#         validation_split=validation_split)

    

    generator = ImageDataGenerator(

        rotation_range=10,

        width_shift_range=.1,

        height_shift_range=.1,

        shear_range=.01,

        zoom_range=.01,

        horizontal_flip=True,

        validation_split=validation_split)

    

    return generator.flow(images, y=labels_categorical, batch_size=batch_size, subset=subset, seed=27)
epochs = 100

batch_size = 4

validation_split = .2

augmented = True
K.clear_session()



model = Sequential()



model.add(Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(64, 3, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(128, 3, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Flatten())



model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dense(25, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



model.summary()



early_stopping = EarlyStopping(

    monitor='val_loss',

    min_delta=0.01,

    patience=3,

    verbose=1,

    mode='min'

)



model_checkpoint = ModelCheckpoint(

    './model.h5',

    monitor='val_loss',

    verbose=1,

    save_best_only=True,

    mode='min'

)



callbacks=[early_stopping, model_checkpoint]



if augmented:

    train_augmented_flow = get_augmented_data_flow(train_images, train_labels_categorical, 

                                                  batch_size, validation_split, 'training')

    

    validation_augmented_flow = get_augmented_data_flow(train_images, train_labels_categorical, 

                                                  batch_size, validation_split, 'validation')

    

    batches = train_images.shape[0] / (batch_size*2)

    steps = ceil(batches * (1-validation_split))

    validation_steps = ceil(batches * validation_split)

    

    history = model.fit_generator(train_augmented_flow, steps_per_epoch=steps,

                                  validation_data=validation_augmented_flow, validation_steps=validation_steps,

                                  epochs=epochs, callbacks=callbacks)

else:

    history = model.fit(train_images, train_labels_categorical, validation_split=validation_split,

                        epochs=epochs, callbacks=callbacks, batch_size=batch_size)



model = load_model('model.h5')
print(history.history)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title("Accuracy")

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train','validation'])

plt.savefig('learning_curve.pdf', bbox_inches='tight')

plt.show()
def evaluate_model(images, labels, labels_categorical):

    start_time = time()

    evaluations = model.evaluate(images, labels_categorical)

    for i in range(len(model.metrics_names)):

        print("{}: {:.2f}%".format(model.metrics_names[i], evaluations[i] * 100))

    print('Took {:.0f} seconds to evaluate this set.'.format(time() - start_time))



    start_time = time()

    predictions = model.predict(images)

    print('Took {:.0f} seconds to get predictions on this set.'.format(time() - start_time))



    y_pred = np.argmax(predictions, axis=1)

    y_true = labels

    return y_pred, y_true
train_pred, train_true = evaluate_model(train_images, train_labels, train_labels_categorical)

test_pred, test_true = evaluate_model(test_images, test_labels, test_labels_categorical)
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    '''

    Plot a confusion matrix heatmap using matplotlib. This code was obtained from

    the scikit-learn documentation:



    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    '''

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return





def plot_confusion_matrix_with_default_options(y_pred, y_true, classes, set_prefix):

    '''Plot a confusion matrix heatmap with a default size and default options.'''

    cm = confusion_matrix(y_true, y_pred)

    with sns.axes_style('ticks'):

        plt.figure(figsize=(16, 16))

        plot_confusion_matrix(cm, classes)

        plt.savefig('{}_confusion_matrix.pdf'.format(set_prefix), bbox_inches='tight')

        plt.show()

    return
print(classification_report(train_pred, train_true, target_names=classes))
with sns.axes_style('ticks'):

    plot_confusion_matrix_with_default_options(train_pred, train_true, classes, 'training')
print(classification_report(test_pred, test_true, target_names=classes))
with sns.axes_style('ticks'):

    plot_confusion_matrix_with_default_options(test_pred, test_true, classes, 'testing')