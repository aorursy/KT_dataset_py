import pandas as pd

import urllib

import matplotlib.pyplot as plt

import numpy as np

import cv2

import glob

import os

import time

from PIL import Image



from keras.applications.vgg16 import VGG16

from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout

from keras.models import Model, Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_json("/kaggle/input/vehicle-number-plate-detection/Indian_Number_plates.json", lines=True)

df.head()
os.mkdir("Indian Number Plates")
dataset = dict()

dataset["image_name"] = list()

dataset["image_width"] = list()

dataset["image_height"] = list()

dataset["top_x"] = list()

dataset["top_y"] = list()

dataset["bottom_x"] = list()

dataset["bottom_y"] = list()



counter = 0

for index, row in df.iterrows():

    img = urllib.request.urlopen(row["content"])

    img = Image.open(img)

    img = img.convert('RGB')

    img.save("Indian Number Plates/licensed_car{}.jpeg".format(counter), "JPEG")

    

    dataset["image_name"].append("licensed_car{}".format(counter))

    

    data = row["annotation"]

    

    dataset["image_width"].append(data[0]["imageWidth"])

    dataset["image_height"].append(data[0]["imageHeight"])

    dataset["top_x"].append(data[0]["points"][0]["x"])

    dataset["top_y"].append(data[0]["points"][0]["y"])

    dataset["bottom_x"].append(data[0]["points"][1]["x"])

    dataset["bottom_y"].append(data[0]["points"][1]["y"])

    

    counter += 1

print("Downloaded {} car images.".format(counter))
df = pd.DataFrame(dataset)

df.head()
df.to_csv("indian_license_plates.csv", index=False)
df = pd.read_csv("indian_license_plates.csv")

df["image_name"] = df["image_name"] + ".jpeg"

df.drop(["image_width", "image_height"], axis=1, inplace=True)

df.head()
lucky_test_samples = np.random.randint(0, len(df), 5)

reduced_df = df.drop(lucky_test_samples, axis=0)
WIDTH = 224

HEIGHT = 224

CHANNEL = 3



def show_img(index):

    image = cv2.imread("Indian Number Plates/" + df["image_name"].iloc[index])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, dsize=(WIDTH, HEIGHT))



    tx = int(df["top_x"].iloc[index] * WIDTH)

    ty = int(df["top_y"].iloc[index] * HEIGHT)

    bx = int(df["bottom_x"].iloc[index] * WIDTH)

    by = int(df["bottom_y"].iloc[index] * HEIGHT)



    image = cv2.rectangle(image, (tx, ty), (bx, by), (0, 0, 255), 1)

    plt.imshow(image)

    plt.show()
show_img(5)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)



train_generator = datagen.flow_from_dataframe(

    reduced_df,

    directory="Indian Number Plates/",

    x_col="image_name",

    y_col=["top_x", "top_y", "bottom_x", "bottom_y"],

    target_size=(WIDTH, HEIGHT),

    batch_size=32, 

    class_mode="other",

    subset="training")



validation_generator = datagen.flow_from_dataframe(

    reduced_df,

    directory="Indian Number Plates/",

    x_col="image_name",

    y_col=["top_x", "top_y", "bottom_x", "bottom_y"],

    target_size=(WIDTH, HEIGHT),

    batch_size=32, 

    class_mode="other",

    subset="validation")
model = Sequential()

model.add(VGG16(weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, CHANNEL)))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dense(64, activation="relu"))

model.add(Dense(64, activation="relu"))

model.add(Dense(4, activation="sigmoid"))



model.layers[-6].trainable = False



model.summary()
STEP_SIZE_TRAIN = int(np.ceil(train_generator.n / train_generator.batch_size))

STEP_SIZE_VAL = int(np.ceil(validation_generator.n / validation_generator.batch_size))



print("Train step size:", STEP_SIZE_TRAIN)

print("Validation step size:", STEP_SIZE_VAL)



train_generator.reset()

validation_generator.reset()
adam = Adam(lr=0.0005)

model.compile(optimizer=adam, loss="mse")
history = model.fit_generator(train_generator,

    steps_per_epoch=STEP_SIZE_TRAIN,

    validation_data=validation_generator,

    validation_steps=STEP_SIZE_VAL,

    epochs=30)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
model.evaluate_generator(validation_generator, steps=STEP_SIZE_VAL)
for idx, row in df.iloc[lucky_test_samples].iterrows():    

    img = cv2.resize(cv2.imread("Indian Number Plates/" + row[0]) / 255.0, dsize=(WIDTH, HEIGHT))

    y_hat = model.predict(img.reshape(1, WIDTH, HEIGHT, 3)).reshape(-1) * WIDTH

    

    xt, yt = y_hat[0], y_hat[1]

    xb, yb = y_hat[2], y_hat[3]

    

    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)

    image = cv2.rectangle(img, (xt, yt), (xb, yb), (0, 0, 255), 1)

    plt.imshow(image)

    plt.show()