import tensorflow as tf

tf.__version__
import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

from PIL import Image

from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import mobilenet, resnet_v2

from tensorflow.keras.models import Model

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import optimizers

from tensorflow.keras.callbacks import ModelCheckpoint



dataset_path = Path("../input/common-mobile-web-app-icons/")
# include in svg file

image_paths = [path for path in dataset_path.glob("*/*.jpg")]
len(image_paths)
[path for path in dataset_path.glob("*/*.svg")]
x_col_name = "image_path"

y_col_name = "class"

df = pd.DataFrame({x_col_name: image_paths})

df[y_col_name] = df[x_col_name].map(lambda x: x.parent.stem)

df[x_col_name] = df[x_col_name].map(lambda x: str(x))
df.head()
# common using mobile app UI labels

USE_LABELS = ['arrow_left', 'notifications', 'play', 'info', 'mail',

              'globe', 'upload', 'music', 'close', 'user', 'settings', 'home',

              'fast_forward', 'trash', 'question', 'map', 'eye', 'check_mark',

              'sort', 'overflow_menu', 'minimize', 'save', 'delete',

              'maximize', 'download', 'share', 'external_link', 'thumbs_up',

              'search', 'arrow_right', 'crop', 'camera', 'refresh', 'add',

              'volume', 'favorite', 'menu', 'edit', 'fab', 'link', 'arrow_up',

              'arrow_down', 'tag', 'warning', 'bookmark', 'cart', 'cloud',

              'filter', '_negative']
labels = set(df[y_col_name].unique()).difference(set(USE_LABELS))

drop_indexes = pd.Index([])

for label in labels:

    drop_index = df[df[y_col_name] == label].index

    drop_indexes = drop_indexes.union(drop_index)
df.drop(index=drop_indexes, inplace=True)
df.shape
test_size = 0.2

random_state = 1234

x_train, x_val, y_train, y_val = train_test_split(df[x_col_name], df[y_col_name],

                                                      test_size=test_size,

                                                      shuffle=True,

                                                      random_state=random_state,

                                                      stratify=df[y_col_name])
num_classes = len(df[y_col_name].unique())

num_classes
def plot_history(history):

    # Plot training & validation accuracy values

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



    # Plot training & validation loss values

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()
def build_model(base_model, n_classes):

#     base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)

    x = Dense(1024, activation="relu")(x)

    x = Dropout(0.25)(x)

    x = Dense(512, activation="relu")(x)

    x = Dropout(0.25)(x)

    y = Dense(n_classes, activation="softmax")(x)



    model = Model(inputs=base_model.input,

                  outputs=y)

    return model
width, height = 224, 224

target_size = (height, width)

num_channels = 3

input_shapes = (height, width, num_channels)

epochs = 10

lr = 0.0001

batch_size = 64

opt = optimizers.Adam(lr=lr)
base_model = resnet_v2.ResNet101V2(include_top=False,

                                   weights='imagenet',

                                   input_shape=input_shapes)
model = build_model(base_model, num_classes)

model.summary()
filepath = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"

callbacks = [ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)]
train_gen = ImageDataGenerator(rotation_range=45,

                               width_shift_range=.15,

                               height_shift_range=.15,

                               horizontal_flip=True,

                               zoom_range=0.5,

                               preprocessing_function=resnet_v2.preprocess_input)

train_generator = train_gen.flow_from_dataframe(

        pd.concat([x_train, y_train],

                  axis=1),

        x_col=x_col_name,

        y_col=y_col_name,

        target_size=target_size,

        batch_size=batch_size,

        class_mode='categorical',

        subset='training')

valid_gen = ImageDataGenerator(preprocessing_function=resnet_v2.preprocess_input)

valid_generator = valid_gen.flow_from_dataframe(pd.concat([x_val, y_val],

                                                              axis=1),

                                                    x_col=x_col_name,

                                                    y_col=y_col_name,

                                                    target_size=target_size,

                                                    batch_size=batch_size,

                                                    class_mode='categorical',

                                                    subset='training')

model.compile(optimizer=opt,

              loss='categorical_crossentropy',

              metrics=['accuracy'])

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size



history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=STEP_SIZE_TRAIN,

                              validation_data=valid_generator,

                              validation_steps=STEP_SIZE_VALID,

                              epochs=epochs,

                              callbacks=callbacks)
plot_history(history)