import numpy as np

import pandas as pd

import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))
train_dir = "../input/dogs-vs-cats/train/train"

test_dir = "../input/dogs-vs-cats/test1/test1"
print(len(os.listdir(train_dir)), len(os.listdir(test_dir)))
filenames = os.listdir(train_dir)

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})

df = df.iloc[:1000]

df.head()
df["category"].value_counts()
df = df.reset_index(drop=True)
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
img_size = 224

batch_size = 20
from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout

from keras.preprocessing.image import ImageDataGenerator

from keras import applications, optimizers
base_model = applications.InceptionResNetV2(weights='imagenet', 

                          include_top=False, 

                          input_shape=(img_size, img_size, 3))
base_model.trainable = False
x = base_model.output

x = Flatten()(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

predictions = Dense(1, activation="sigmoid")(x)

model = Model(input = base_model.input, output = predictions)



model.compile(loss='binary_crossentropy', optimizer = optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])#optimizers.rmsprop(lr=0.0001, decay=1e-5)
from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical



train_datagen=ImageDataGenerator(

    rotation_range=40,

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip = True,

    width_shift_range=0.2,

    height_shift_range=0.2

)



train_generator = train_datagen.flow_from_dataframe(

    df, 

    train_dir, 

    x_col = 'filename',

    y_col = 'category',

    class_mode = 'binary',

    target_size = (img_size, img_size),

    batch_size = batch_size

)
%%time

history = model.fit_generator(

    train_generator,

    steps_per_epoch=100,

    epochs=30,

    verbose=1

)
history.history
import matplotlib.pyplot as plt

show_history=pd.DataFrame()

show_history["Train Loss"]=history.history['loss']

show_history["Train Accuracy"]=history.history['acc']

show_history.plot(figsize=(12,8))

plt.title("Convulutional Model Train Loss and Accuracy History")

plt.show()
test_files = os.listdir(test_dir)

test_df = pd.DataFrame({

    'filename': test_files

})

nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    test_dir, 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=(img_size, img_size),

    batch_size=batch_size,

    shuffle=False

)
%%time

test_generator.reset()

predict = model.predict_generator(test_generator,

                                  steps=np.ceil(nb_samples/batch_size))

                                  #steps = len(test_generator.filenames))

len(predict)
threshold = 0.5

test_df['category'] = np.where(predict > threshold, 'dog', 'cat')
sample_test = test_df.sample(n=18).reset_index()

plt.figure(figsize=(12, 24))

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img(test_dir+"/"+filename, 

                   target_size=(img_size, img_size))

    plt.subplot(6, 3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + ' (' + "{}".format(category) + ')')

plt.tight_layout()

plt.show()
test_df['category'].value_counts()
test_df["category"] = test_df["category"].replace({'cat': 0, 'dog': 1})
test_df = test_df.rename(columns = {'filename':'id', 'category':'label'})

test_df['id'] = test_df['id'].str.split('.').str[0]
test_df.to_csv('submission.csv', index=False)