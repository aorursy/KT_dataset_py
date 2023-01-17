import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
df.head(10)
df.tail(10)
df.sample(10)
df.describe()
breeds = df.breed.unique()

breeds
len(breeds)
row = df.iloc[0]

img = cv2.imread("/kaggle/input/dog-breed-identification/train/"+row.id+".jpg")

plt.imshow(img)

plt.title(row.breed)
print(img.shape)

IMAGE_WIDTH = img.shape[0]

IMAGE_HEIGHT = img.shape[1]

IMAGE_CHANNEL = img.shape[2]
df["filename"] = df['id'] + ".jpg"

df.head()
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()



model.add(Conv2D(64, kernel_size=(1, 1), activation="relu", input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL)))

model.add(MaxPooling2D())



model.add(Conv2D(32, kernel_size=(1, 1), activation="relu"))

model.add(MaxPooling2D())



model.add(Flatten())

model.add(Dense(512, activation="relu"))

model.add(Dense(120, activation="softmax"))



model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# mini_df = df.groupby('breed').apply(lambda s: s.sample(50))
# mini_df.head()
train_df, valid_df = train_test_split(df, test_size=0.15, random_state=42)
train_gen = ImageDataGenerator(

    rescale=1.0/255,

    rotation_range=15,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True

)

train_generator = train_gen.flow_from_dataframe(

    train_df,

    directory="/kaggle/input/dog-breed-identification/train", 

    x_col="filename",

    y_col="breed",

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

    class_mode="categorical"

)
valid_gen = ImageDataGenerator(

    rescale=1.0/255,

    rotation_range=15,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True

)

valid_generator = train_gen.flow_from_dataframe(

    valid_df,

    directory="/kaggle/input/dog-breed-identification/train", 

    x_col="filename",

    y_col="breed",

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

    class_mode="categorical"

)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(patience=2, verbose=1, factor=0.5, min_lr=0.00001)
model.fit_generator(train_generator, validation_data=valid_generator, epochs=1000, callbacks = [earlystop, learning_rate_reduction])
scores = model.evaluate_generator(valid_generator)

print("Loss = ", scores[0])

print("Accuracy = ", scores[1])
model.save_weights("model.h5")
test_df = pd.read_csv('/kaggle/input/dog-breed-identification/sample_submission.csv')
test_df.head()
test_df["filename"] = test_df['id'] + ".jpg"
# mini_test_df = test_df.sample(300).reset_index()
test_gen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_gen.flow_from_dataframe(

    test_df,

    directory="/kaggle/input/dog-breed-identification/test", 

    x_col="filename",

    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),

    class_mode=None

)
predictions = model.predict_generator(test_generator)
labels = list(train_generator.class_indices.keys())

columns = ['id'] + labels

submission = pd.DataFrame(columns=columns)

submission['id'] = mini_test_df.id

submission[labels] = predictions
submission.head()