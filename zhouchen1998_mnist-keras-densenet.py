import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
!mkdir data

!mkdir data/train

!mkdir data/test
import cv2

from tqdm import tqdm

df_train_processed = pd.DataFrame(columns=['id', 'label'])



ids = []

for i in tqdm(range(len(df_train))):

    img = df_train.iloc[i, 1:].values.reshape(28, 28)

    img = cv2.imwrite('data/train/{}.png'.format(i), img)

    ids.append('{}.png'.format(i))

df_train_processed['id'] = ids

df_train_processed['label'] = df_train['label'].astype(str)
plt.figure(figsize=(16, 8))

df_train_processed['label'].value_counts().plot(kind='pie')
import cv2

from tqdm import tqdm

df_test_processed = pd.DataFrame(columns=['id', 'label'])



ids = []

for i in tqdm(range(len(df_test))):

    img = df_test.iloc[i, :].values.reshape(28, 28)

    img = cv2.imwrite('data/test/{}.png'.format(i), img)

    ids.append('{}.png'.format(i))

df_test_processed['id'] = ids
from keras.preprocessing.image import ImageDataGenerator



train_gen = ImageDataGenerator(rescale=1/255., validation_split=0.2, horizontal_flip=False, height_shift_range=0.2, width_shift_range=0.1, rotation_range=5)

test_gen = ImageDataGenerator(rescale=1/255.) 

img_size = (224, 224)

batch_size = 64

train_generator = train_gen.flow_from_dataframe(dataframe=df_train_processed,

                                               directory='data/train',

                                               x_col='id',

                                               y_col='label',

                                               batch_size=batch_size,

                                               class_mode='categorical',

                                               target_size=img_size, 

                                               subset='training')

valid_generator = train_gen.flow_from_dataframe(dataframe=df_train_processed,

                                                  directory="data/train",

                                                  x_col="id",

                                                  y_col="label",

                                                  batch_size=batch_size,

                                                  class_mode="categorical",    

                                                  target_size=img_size,

                                                  subset='validation')

test_generator = test_gen.flow_from_dataframe(dataframe=df_test_processed,

                                                  directory = "data/test",

                                                  x_col="id",

                                                  target_size=img_size,

                                                  batch_size=1,

                                                  shuffle=False,

                                                  class_mode=None)
from keras.applications.densenet import DenseNet121

from keras.models import Model

from keras.layers import GlobalAveragePooling2D, Input, Dropout, Dense, BatchNormalization

from keras.optimizers import Adam



def build_densenet(input_shape=(224, 224, 3), n_classes=10):

    input_layer = Input(shape=input_shape)

    densenet121 = DenseNet121(include_top=False, weights='imagenet', input_tensor=input_layer)

    x = GlobalAveragePooling2D()(densenet121.output)

    x = Dropout(0.5)(x)

    x = Dense(n_classes, activation='softmax')(x)

    

    model = Model(input_layer, x)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=3e-4), metrics=['accuracy'])

    return model



densenet = build_densenet()
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# 设置超参

epochs = 50

callbacks = [

    ModelCheckpoint('model_best_weights.h5', monitor='val_loss', verbose=True, save_best_only=True, save_weights_only=True),

    ReduceLROnPlateau(monitor='val_loss', patience=5, min_lr=1e-6),

    EarlyStopping(monitor='val_acc', patience=20)

]
history = densenet.fit_generator(train_generator, 

                              steps_per_epoch=train_generator.n//train_generator.batch_size, 

                              validation_data=valid_generator, 

                              validation_steps=valid_generator.n//valid_generator.batch_size, 

                              epochs=epochs, 

                              callbacks=callbacks

                             )
pred_densenet = densenet.predict_generator(test_generator, test_generator.n, verbose=1)

pred = np.argmax(pred_densenet, axis=1)
df_test = pd.read_csv('../input/sample_submission.csv')

df_test['Label'] = pred

df_test.to_csv('submission.csv', index=False)

df_test.head()

!rm -rf data/