import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow.keras.layers as L

from tensorflow.keras import Model

from sklearn.metrics import f1_score

from tensorflow.keras import callbacks
train = pd.read_csv("../input/liverpool-ion-switching/train.csv")

test = pd.read_csv("../input/liverpool-ion-switching/test.csv")

sub = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv", dtype=dict(time=str))
n_classes = train.open_channels.unique().shape[0]
seq_len = 1000



X = train.signal.values.reshape(-1, seq_len, 1)

y = train.open_channels.values.reshape(-1, seq_len, 1)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)



X_test = test.signal.values.reshape(-1, seq_len, 1)
def build_model(n_classes, seq_len=500, n_units=256):

    inputs = L.Input(shape=(seq_len, 1))

    x = L.Dense(n_units, activation='linear')(inputs)

    

    x = L.Bidirectional(L.GRU(n_units, return_sequences=True))(x)

    x = L.Bidirectional(L.GRU(n_units, return_sequences=True))(x)

    x = L.Dense(n_classes, activation='softmax')(x)

    

    model = Model(inputs=inputs, outputs=x)

    model.compile('adam', loss='sparse_categorical_crossentropy')

    

    return model
model = build_model(n_classes, seq_len)

model.summary()
class F1Callback(callbacks.Callback):

    def __init__(self, X_val, y_val):

        super().__init__()

        self.X = X_val

        self.y = y_val.reshape(-1)

    def on_epoch_begin(self, epoch, logs=None):

        if epoch == 0:

            return

        pred = (

            model

            .predict(self.X, batch_size=64)

            .argmax(axis=-1)

            .reshape(-1)

        )

        

        score = f1_score(self.y, pred, average='macro')

        

        print(f"val_f1_macro: {score:.4f}")
model.fit(

    X_train, y_train, 

    batch_size=64,

    epochs=30,

    callbacks=[

        callbacks.ReduceLROnPlateau(),

        F1Callback(X_valid, y_valid),

        callbacks.ModelCheckpoint('model.h5')

    ],

    validation_data=(X_valid, y_valid)

)
model.load_weights('model.h5')

valid_pred = model.predict(X_valid, batch_size=64).argmax(axis=-1)

f1_score(y_valid.reshape(-1), valid_pred.reshape(-1), average='macro')
test_pred = model.predict(X_test, batch_size=64).argmax(axis=-1)

sub.open_channels = test_pred.reshape(-1)

sub.to_csv('submission.csv', index=False)