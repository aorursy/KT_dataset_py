%ls ../input/dog-breed-identification/
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("../input/dog-breed-identification/")
TRAIN_ROOT_PATH = INPUT_PATH / "train"
TEST_ROOT_PATH = INPUT_PATH / "test"
LOG_PATH = Path("./logs")
CP_PATH = LOG_PATH / "cp.ckpt"
labels_df = pd.read_csv(INPUT_PATH / "labels.csv")

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
# labels.csv
# col_name
image_id_col = "id"
label_col = "breed"

# training settings
EPOCHS = 10
BATCH_SIZE = 32
IMAGE_SIZE = 224
CHANNELS = 3
RANDOM_STATE = 1234
SEED = 5678
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
NUM_CLASSES = labels_df[label_col].nunique()

# tf.data.Dataset settings
AUTOTUNE = tf.data.experimental.AUTOTUNE
labels_df.head()
label_to_index = dict((name, index) for index, name in enumerate(labels_df[label_col].unique()))
label_to_index
# Reference
# https://www.tensorflow.org/tutorials/load_data/images
def preprocess_image(image, size=224, channels=3):
    image = tf.image.decode_jpeg(image, channels=channels)
    image = tf.image.resize(image, [size, size])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def get_label_index(label_name):
    return label_to_index.get(label_name)


def get_image_path(part_path, ext=".jpg"):
    file_name = part_path + ext
    return str(TRAIN_ROOT_PATH / file_name)


def get_img_id(img_path):
    return img_path.split("/")[-1].strip(".jpg")


def get_label(img_path):
    return labels_df[labels_df["id"] == get_img_id(img_path)][label_col]
train_image_paths = list(map(str, (INPUT_PATH / "train").glob("*.jpg")))
test_image_paths = list(map(str, (INPUT_PATH / "test").glob("*.jpg")))
x_train, x_valid, y_train, y_valid = train_test_split(labels_df[image_id_col],
                                                      labels_df[label_col], 
                                                      stratify=labels_df[label_col],
                                                      test_size=0.2,
                                                      random_state=RANDOM_STATE)
x_train.shape, x_valid.shape, y_train.shape, y_valid.shape
get_image_path(x_train.values[0])
get_label_index(y_train.values[0])
# TFRecord

# train_image_ds = tf.data.Dataset.from_tensor_slices(train_image_paths).map(tf.io.read_file)
# test_image_ds = tf.data.Dataset.from_tensor_slices(test_image_paths).map(tf.io.read_file)
# train_paths_ds = tf.data.Dataset.from_tensor_slices(train_image_paths)
# test_paths_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
# train_image_ds = train_paths_ds.map(load_and_preprocess_image)
# test_image_ds = test_paths_ds.map(load_and_preprocess_image)
# train_tfrec = tf.data.experimental.TFRecordWriter('train_images.tfrec')
# test_tfrec = tf.data.experimental.TFRecordWriter('test_images.tfrec')
# train_image_ds = train_image_ds.map(tf.io.serialize_tensor)
# test_image_ds = test_image_ds.map(tf.io.serialize_tensor)
# train_tfrec.write(train_image_ds)
# test_tfrec.write(test_image_ds)
# path_ds = tf.data.Dataset.from_tensor_slices(list(map(get_image_path, x_train.values)))
# image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
# label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(list(map(get_label_index, y_train.values)),
#                                                       tf.int64))

def get_pair_ds(_images, _labels):
    path_ds = tf.data.Dataset.from_tensor_slices(list(map(get_image_path, _images.values)))
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(list(map(get_label_index, _labels.values)),
                                                          tf.int64))
    return image_ds, label_ds


def apply_ds(_length, _image_ds, _label_ds):
    ds = tf.data.Dataset.zip((_image_ds, _label_ds))
    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=_length))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = apply_ds(len(x_train), *get_pair_ds(x_train, y_train))
valid_ds = apply_ds(len(x_valid), *get_pair_ds(x_valid, y_valid))
# Reference
# https://tfhub.dev/google/collections/efficientnet/1
feature_extractor_url = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"

def building_model(_input_shape, _num_classes):
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                             input_shape=_input_shape)
    feature_extractor_layer.trainable = False
    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(_num_classes, activation='softmax')
    ])
    return model
model = building_model(INPUT_SHAPE, NUM_CLASSES)
model.summary()
# callbacks
tensorboard = tf.keras.callbacks.TensorBoard(str(LOG_PATH))
cp_callback = tf.keras.callbacks.ModelCheckpoint(str(CP_PATH), 
                                                 save_weights_only=True,
                                                 verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  patience=3)
_callbacks = [tensorboard, early_stopping, cp_callback]
tr_steps_per_epoch = tf.math.ceil(len(x_train) / BATCH_SIZE).numpy()
va_steps_per_epoch = tf.math.ceil(len(x_valid) / BATCH_SIZE).numpy()
tr_steps_per_epoch, va_steps_per_epoch
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['acc'])

history = model.fit(train_ds,
                    validation_data=valid_ds,
                    epochs=EPOCHS,
                    steps_per_epoch=tr_steps_per_epoch,
                    validation_steps=va_steps_per_epoch,
                    callbacks=_callbacks)
# loss, acc, val_loss, val_acc 
hist_keys = list(history.history.keys())