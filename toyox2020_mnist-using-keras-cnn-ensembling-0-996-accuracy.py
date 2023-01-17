import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
# divide training data into features and labels

X_train = train.iloc[:,1:]

y_train = train.iloc[:,0]
# Normalize and reshape image

X_train = X_train.values.reshape(-1, 28, 28, 1)/255.

test = test.values.reshape(-1, 28, 28, 1)/255.

# One Hot encoding the label

y_train = to_categorical(y_train, 10)
datagen = ImageDataGenerator(

            rotation_range=15,

            width_shift_range=0.1,

            height_shift_range=0.1,

            zoom_range=0.1,

            shear_range=0.2

            )
def create_model():

    model = Sequential()



    model.add(Conv2D(32, (3,3), padding='same', input_shape=X_train.shape[1:], activation='relu'))

    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))

    model.add(Conv2D(32, (5,5), padding='same', activation='relu'))

    #model.add(MaxPool2D(2,2))



    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))

    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))

    model.add(Conv2D(64, (5,5), strides=2, padding='same', activation='relu'))

    #model.add(MaxPool2D(2,2))



    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model
EPOCHS = 30

BATCH_SIZE = 50

ENSEMBLES = 5 # number of models to ensemble

result_list = [] # store results for correlation matrix

histories = [] # store histories for training and validation curves

results = np.zeros((test.shape[0],10))



callback_list = [

    ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=2),

    EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=4)

]



for i in range(ENSEMBLES):

    # split training and validation sets

    X_train_tmp, X_val, y_train_tmp, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=i)

    # create model

    model = create_model()

    # fit the model

    print('training No.', i)

    history = model.fit(datagen.flow(X_train_tmp, y_train_tmp, batch_size=BATCH_SIZE),

                   verbose=0,

                   epochs=EPOCHS,

                   callbacks=callback_list,

                   validation_data=(X_val, y_val),

                   steps_per_epoch=X_train_tmp.shape[0] // BATCH_SIZE)

    # save results

    histories.append(history)

    result = model.predict(test)

    results += result

    result_list.append(result)
# check correlation of each predictions

corr_preds = pd.DataFrame([np.argmax(result, axis=1) for result in result_list]).T.corr()

fig = sns.heatmap(corr_preds, annot=True, fmt='.3f', cmap='rainbow')

fig.set_title('Predictions correlation matrix', fontsize=16, y=1.05)

plt.show()
fig, ax = plt.subplots(2, 1, figsize=(12,6))



for e in range(ENSEMBLES):

    loss = histories[e].history['loss']

    val_loss = histories[e].history['val_loss']

    acc = histories[e].history['accuracy']

    val_acc = histories[e].history['val_accuracy']

    ax[0].set_title('loss')

    ax[0].plot(loss, 'b', linewidth=1)

    ax[0].plot(val_loss, 'r', linewidth=1)

    ax[0].grid(color='black', linestyle='-', linewidth=0.2)

    ax[1].set_title('accuracy')

    ax[1].plot(acc, 'b', linewidth=1)

    ax[1].plot(val_acc, 'r', linewidth=1)

    ax[1].grid(color='black', linestyle='-', linewidth=0.2)

    

ax[0].legend(['Training loss', 'Validation loss'], shadow=True)     

ax[1].legend(['Training accuracy', 'Validation accuracy'], shadow=True)



plt.tight_layout()

plt.show()
results = np.argmax(results, axis=1)

results = pd.Series(results, name='Label')

submission = pd.concat([pd.Series(range(1,28001), name='ImageID'), results], axis=1)

submission.to_csv('submission.csv', index=False)