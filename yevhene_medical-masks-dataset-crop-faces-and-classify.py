import tensorflow as tf

import numpy as np

import os

import glob

import math

import matplotlib.pyplot as plt

import random

from xml.etree import ElementTree

from PIL import Image, ImageOps, ImageDraw



from tensorflow.keras.utils import Progbar

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, GlobalMaxPooling2D, Input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import applications



%matplotlib inline
print(tf.__version__)
random_seed = 42



img_height = 64

img_width = 64

img_channels = 3



np.random.seed(random_seed)

tf.random.set_seed(random_seed)

random.seed(random_seed)
images_dir = "/kaggle/input/medical-masks/images"

meta_dir = "/kaggle/input/medical-masks/labels"

data_dir = "./source"

train_dir = os.path.join(data_dir, "train")

test_dir = os.path.join(data_dir, "test")
!rm -rf {data_dir}

!mkdir {data_dir}
def random_image(images_dir):

    files = glob.glob(os.path.join(images_dir, "*"), recursive=True)

    return random.choice(files)



image_path = random_image(images_dir)

image_path
def load_image(image_path):

    return Image.open(image_path).convert("RGB")



image = load_image(image_path)

plt.imshow(image)
def image_meta_path(image_path, meta_dir):

    image_filename = os.path.basename(image_path)

    basename = os.path.splitext(image_filename)[0]

    return os.path.join(meta_dir, f"{basename}.xml")



image_meta_path(image_path, meta_dir)
def get_regions_meta(image_path):

    meta_file = image_meta_path(image_path, meta_dir)

    root = ElementTree.parse(meta_file).getroot()

    regions = []

    for object_tag in root.findall("object"):

        name = object_tag.find("name").text

        xmin = int(object_tag.find("bndbox/xmin").text)

        xmax = int(object_tag.find("bndbox/xmax").text)

        ymin = int(object_tag.find("bndbox/ymin").text)

        ymax = int(object_tag.find("bndbox/ymax").text)

        regions.append({ "name": name, "coordinates": (xmin, ymin, xmax, ymax) })

    return regions



regions_meta = get_regions_meta(image_path)

regions_meta
def process_image(image, size=None, crop=None):

    if not crop is None:

        image = image.crop(crop)

    if not size is None:

        image = image.resize(size)

    return image



def crop_regions(image, regions_meta, size=None):

    return list(map(lambda region_meta: (region_meta["name"], process_image(image, crop=region_meta["coordinates"], size=size)), regions_meta))



regions = crop_regions(image, regions_meta, size=(img_width, img_height))



def plot_regions(regions):

    fig = plt.figure(figsize=(10, 10))

    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    columns = 4

    rows = math.ceil(len(regions) / columns)

    

    for i, region in enumerate(regions):

        label, image = region

        ax = fig.add_subplot(rows, columns, i + 1)

        ax.set_title(label)

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

        plt.imshow(image)

        

    plt.show()



plot_regions(regions)
test_split = 0.2



def image_dir(label):

    if not label in ['good', 'bad']:

        return None



    if random.random() > test_split:

        split_dir = train_dir

    else:

        split_dir = test_dir

  

    return os.path.join(split_dir, label)



image_dir('good')
def prepare_image(image_path, image_index, size=None):

    regions_meta = get_regions_meta(image_path)

    image = load_image(image_path)

    regions = crop_regions(image, regions_meta, size=size)

    for region_index, region in enumerate(regions):

        label, image = region

        region_dir = image_dir(label)

        if region_dir == None:

            continue

        if not os.path.isdir(region_dir):

            os.makedirs(region_dir)

        region_filename = f"{image_index}_{region_index}.jpg"

        image.save(os.path.join(region_dir, region_filename), "JPEG")
def prepare_images(images_dir, size=None):

    images_paths = glob.glob(os.path.join(images_dir, '*'), recursive=True)

    progbar = Progbar(target = len(images_paths))

    progbar.update(0)

    for image_index, image_path in enumerate(images_paths):

        prepare_image(image_path, image_index, size=size)

        progbar.update(image_index + 1)



prepare_images(images_dir, size=(img_width, img_height))
!ls -R | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\//--/g' -e 's/^/   /' -e 's/-/|/'
batch_size = 16



train_data = ImageDataGenerator(

    rescale=1./255,

    rotation_range=15,

    width_shift_range=0.3,

    height_shift_range=0.3,

    brightness_range=(0.67, 1.0),

    shear_range=0.3,

    zoom_range=0.3,

    horizontal_flip=True,

    fill_mode="nearest"

).flow_from_directory(

    batch_size=batch_size,

    directory=train_dir,

    shuffle=True,

    target_size=(img_height, img_width), 

    class_mode="binary"

)



validation_data = ImageDataGenerator(

    rescale=1./255

).flow_from_directory(

    batch_size=batch_size,

    directory=test_dir,

    shuffle=True,

    target_size=(img_height, img_width), 

    class_mode="binary"

)
def analyze(history):

    best_accuracy = np.max(history.history["val_accuracy"])

    best_loss = np.min(history.history["val_loss"])

    print(f"Best accuracy: {best_accuracy}")

    print(f"Best logloss: {best_loss}")

    

    acc = history.history["accuracy"]

    val_acc = history.history["val_accuracy"]



    loss = history.history["loss"]

    val_loss = history.history["val_loss"]

    

    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)

    plt.plot(acc, label="Training Accuracy")

    plt.plot(val_acc, label="Validation Accuracy")

    plt.plot(np.argmax(history.history["val_accuracy"]), best_accuracy, "o", color="green")

    plt.legend(loc="lower right")

    plt.ylabel("Accuracy")

    plt.ylim([min(plt.ylim()),1])

    plt.title("Training and Validation Accuracy")



    plt.subplot(2, 1, 2)

    plt.plot(loss, label="Training Loss")

    plt.plot(val_loss, label="Validation Loss")

    plt.plot(np.argmin(history.history["val_loss"]), best_loss, "o", color="green")

    plt.legend(loc="upper right")

    plt.ylabel("Cross Entropy")

    plt.ylim([min(plt.ylim()),max(plt.ylim())])

    plt.title("Training and Validation Loss")

    plt.xlabel("epoch")

    plt.show()
epochs = 50

verbose = 0
plain_model = Sequential([

    Input((img_height, img_width, img_channels)),

    Flatten(),

    Dense(16),

    Dense(1)

])



plain_model.summary()
train_data.reset()

validation_data.reset()



plain_model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)



plain_model_history = plain_model.fit(

    train_data,

    epochs=epochs,

    steps_per_epoch=len(train_data),

    validation_data=validation_data,

    validation_steps=len(validation_data),

    verbose=verbose

)
analyze(plain_model_history)
base_transfer_model = applications.Xception(include_top=False, weights="imagenet")

base_transfer_model.trainable = False



transfer_model = Sequential([

    Input((img_height, img_width, img_channels)),

    base_transfer_model,

    GlobalMaxPooling2D(),

    Dense(64),

    Dense(1)

])



transfer_model.summary()
train_data.reset()

validation_data.reset()



transfer_model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)



transfer_model_history = transfer_model.fit(

    train_data,

    epochs=epochs,

    steps_per_epoch=len(train_data),

    validation_data=validation_data,

    validation_steps=len(validation_data),

    verbose=verbose

)
analyze(transfer_model_history)
custom_model = Sequential([

    Input((img_height, img_width, img_channels)),

    Conv2D(32, (2, 2), activation="relu", padding="same"),

    MaxPooling2D(2, 2),

    Conv2D(64, (2, 2), activation="relu", padding="same"),

    MaxPooling2D(2, 2),

    Conv2D(64, (2, 2), activation="relu", padding="same"),

    MaxPooling2D(2, 2),

    Conv2D(64, (2, 2), activation="relu", padding="same"),

    MaxPooling2D(2, 2),

    Flatten(),

    Dropout(0.5),

    Dense(128, activation="relu"),

    Dense(32, activation="relu"),

    Dense(1, activation="sigmoid")

])



custom_model.summary()
train_data.reset()

validation_data.reset()



custom_model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)



custom_model_history = custom_model.fit(

    train_data,

    epochs=epochs,

    steps_per_epoch=len(train_data),

    validation_data=validation_data,

    validation_steps=len(validation_data),

    verbose=verbose

)
analyze(custom_model_history)
plain_model_acc = plain_model_history.history["val_accuracy"]

transfer_model_acc = transfer_model_history.history["val_accuracy"]

custom_model_acc = custom_model_history.history["val_accuracy"]



plt.plot(plain_model_acc, label="Plain Model Accuracy")

plt.plot(transfer_model_acc, label="Transfer Model Accuracy")

plt.plot(custom_model_acc, label="Custom Model Accuracy")

plt.ylabel("Accuracy")

plt.ylim([min(plt.ylim()),1])

plt.title("Model comparison")