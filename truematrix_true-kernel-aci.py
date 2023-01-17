import pandas as pd

import os

os.chdir("../input")
meta_data = pd.read_csv("train.csv")

meta_data.head()
train_dir = "train/train"

test_dir =  "test/test"

os.listdir(train_dir)[:5]

print(len(os.listdir(train_dir)))
from tensorflow.keras.preprocessing.image import ImageDataGenerator



#######Wiithout data augumentation, just for experimental purposes...#########



train_gen = ImageDataGenerator(rescale = 1/255,

                              horizontal_flip = True,

                              height_shift_range = 0.2,

                              width_shift_range = 0.2,

                              brightness_range = [0.2,1.2],)

                              #fill_mode="nearest")

valid_gen = ImageDataGenerator(rescale=1/255)



meta_data.has_cactus = meta_data.has_cactus.astype(str)



train_generator = train_gen.flow_from_dataframe(

    dataframe = meta_data[:15000],

    target_size=(150,150),

    directory = train_dir,

    x_col="id",

    y_col="has_cactus",

    class_mode = "binary"

)



valid_generator = valid_gen.flow_from_dataframe(

    dataframe = meta_data[15000:],

    target_size = (150,150),

    directory = train_dir,

    x_col = "id",

    y_col = "has_cactus",

    class_mode = "binary",

)

from tensorflow import keras



base_model = keras.applications.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(150,150,3), pooling=None, classes=1)
# VGG19 is a state of the art model which has been trained on imagenet dataset

# For our purposes we select the input shape as (32x32x3), here is the summary of the model.

# We can see there are 5 blocks of convolution, and pooling layers.

base_model.summary()
#locking all the layers to prevent their training, as we only want to extend it by adding our own Dense layer classifier.



for layer in base_model.layers:

    layer.trainable = False

    

#Extracting last layer, and collecting it's last output, which we will use to feed into our extended model.



last_layer = base_model.layers[-1]

last_output = last_layer.output



#Adding out own extended version of the model. For simplicity lets take it to 512 neurons in FC layer. And 2 

#neurons in the last layer for classification purpose.



extend = keras.layers.Flatten()(last_output)

extend = keras.layers.Dense(512, activation = "relu") (extend)

extend = keras.layers.Dropout(0.4)(extend)

extend = keras.layers.Dense(1, activation="sigmoid")(extend)



#Defining our extended model now



model = keras.models.Model(base_model.input, extend)



#All looks good, let's compile our model now. We'll use loss as categorical_crossentropy, and optimizer as adam



model.compile(loss = "binary_crossentropy",

             optimizer=keras.optimizers.Adam(lr = 1e-3),

             metrics=["acc"])



model.summary()


model.fit_generator(

    train_generator,

    validation_data = valid_generator,

    verbose = 1,

    shuffle=True,

    epochs = 10,

)
history = model.history
acc = history.history["acc"]

loss = history.history["loss"]

val_acc = history.history["val_acc"]

val_loss = history.history["val_loss"]

epochs = range(len(acc))
import matplotlib.pyplot as plt





plt.plot(epochs, acc, label="Training Accuracy")

plt.plot(epochs, val_acc, label="Validation Accuracy")

plt.axis([0, 4, 0.7, 1])

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.figure()
#model.save("my_model.h5")
os.listdir("test/test")[:5]
import cv2

images = []



for image in os.listdir("test/test"):

    images.append( cv2.imread("test/test/" + image))

import numpy as np

image = np.asarray(images)
image.resize(4000, 32, 32, 3)

image.shape
prediction = model.predict(image)

prediction.resize(4000)
sub = pd.DataFrame({"id" : os.listdir(test_dir),

                   "has_cactus" : prediction})
sub.head()
sub.to_csv("../working/samplesubmission.csv", index=False)