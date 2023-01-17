import numpy as np

import pandas as pd



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv').values
from keras.utils.np_utils import to_categorical

Y_train = train['label'].values

Y_train = to_categorical(Y_train, 10)



train = train.iloc[:,1:].values
X_train = train.reshape(train.shape[0],28,28,1)

test = test.reshape(test.shape[0],28,28,1)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, BatchNormalization, Lambda, Flatten

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization



def Model(n_classes):

    model = Sequential()

    

    model.add(Lambda(lambda x: x/255. - .5, input_shape=(28, 28, 1)))

    

    model.add(Conv2D(30, (5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Conv2D(15, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Flatten())

    

    model.add(Dense(128))

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    

    model.add(Dense(50))

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    

    model.add(Dense(n_classes))

    model.add(Activation('softmax'))

    

    return model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping



EPOCHS = 5

BATCH_SIZE = 1000



model = Model(10)



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                  patience=5, min_lr=0.001)

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)



# Compile and Run

model.compile('adam', "categorical_crossentropy", ["accuracy"])

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, validation_split=0.2, callbacks=[reduce_lr, earlyStopping])
idx = np.arange(1, test.shape[0]+1, 1)

predictions = model.predict_classes(test, verbose=0).flatten()

submission = pd.DataFrame({"ImageId": idx, "Label": predictions})

submission.to_csv('benchmark.csv', index=False)