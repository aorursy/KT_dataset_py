import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Concatenate

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.callbacks import ModelCheckpoint
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train_data = train.text.values

train_labels = train.target.values

test_data = test.text.values
%%time

module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'

embed = hub.KerasLayer(module_url, trainable=True, name='USE_embedding')
def build_model(embed):

    

    model = Sequential([

        Input(shape=[], dtype=tf.string),

        embed,

        Dense(1, activation='sigmoid')

    ])

    model.compile(Adam(2e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
model = build_model(embed)

model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True)



train_history = model.fit(

    train_data, train_labels,

    validation_split=0.25,

    epochs=4,

    callbacks=[checkpoint],

    batch_size=32

)
model.load_weights('model.h5')

test_pred = model.predict(test_data)
submission['target'] = test_pred.round().astype(int)

submission.to_csv('submission.csv', index=False)