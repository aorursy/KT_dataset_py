import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import math



def visualize_labels(df, params, hw=(1, 1), figsize=(20, 10), vertical=True):

    def next_index(h, w, ch, cw):

        return ((ch + math.floor((cw + 1)/w)) % h), ((cw + 1) % w)

    

    x, y = 0, 0

    h, w = hw[0], hw[1]

    fig, all_ax = plt.subplots(h, w, figsize=figsize)



    for param in params:

        ax = all_ax[x, y]

        

        df[param] = df[param].apply(lambda s: "NA" if s is np.nan else s)

        counts = df.groupby([param]).count().reset_index()        

        counts = counts.sort_values(by=['id'])

        labels = counts[param].unique()

        sum = counts.id.sum()

        

        prev_sum=0

        

        if vertical:

            ax.bar(labels, counts.id, align='center')

            ax.set_xticks(labels)

            ax.set_xticklabels(labels)

            ax.set_ylabel(param)

            ax.set_title('%s distribution' % param)

                        

            for i, v in enumerate(counts.id):

                prev_sum = prev_sum + v

                ax.text(i-.25, v, str("%.2f%s" % (1-(v*1/sum), "%")), color='black', fontweight='bold')

                ax.text(i-.25, v+(sum*.03), str(v), color='black', fontweight='bold')

        else:

            ax.barh(labels, counts.id, align='center')

            ax.set_yticks(labels)

            ax.set_yticklabels(labels)

            ax.set_xlabel(param)

            ax.set_title('%s distribution' % param)

                        

            for i, v in enumerate(counts.id):

                prev_sum = prev_sum + v

                ax.text(v+3, i-.25, str("%.2f%s | %s" % (1-(v*1/sum), "%", v)), color='black', fontweight='bold')

        

        x, y = next_index(h, w, x, y)

        

    plt.show()
#### import numpy as np

import pandas as pd



INPUT_IMAGES_DIR = '/kaggle/input/fashion-product-images-dataset/fashion-dataset/images'

INPUT_STYLES = '/kaggle/input/fashion-product-images-dataset/fashion-dataset/styles.csv'



df = pd.read_csv(INPUT_STYLES, error_bad_lines=False)

df['season'] = df.season.apply(lambda row: 'NA' if row is np.nan else row)

df['image'] = df.id.apply(lambda row: "%s.jpg" % row)



train_df = df[:14000][df['season'] != 'NA']

validate_df = df[14001:19000]

test_df = df[19001:20000]
visualize_labels(test_df, ['subCategory', 'articleType', 'baseColour'], hw=(2, 2), figsize=(20, 50), vertical=False)
visualize_labels(train_df, ['gender', 'masterCategory', 'season', 'usage'], hw=(2, 2))

# visualize_labels(validate_df, ['gender', 'masterCategory', 'season', 'usage'], hw=(2, 2))

# visualize_labels(test_df, ['gender', 'masterCategory', 'season', 'usage'], hw=(2, 2))
from keras.preprocessing.image import ImageDataGenerator

HEIGHT = 800

WIDTH = 600

datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()



train_generator=datagen.flow_from_dataframe(

                    dataframe=train_df,

                    directory=INPUT_IMAGES_DIR,

                    x_col="image",

                    y_col="season",

                    batch_size=32,

                    seed=42,

                    shuffle=True,

                    class_mode="categorical",

                    classes=["Spring", "Winter", "Fall", "Summer"],

                    target_size=(HEIGHT,WIDTH))



valid_generator=test_datagen.flow_from_dataframe(

                    dataframe=validate_df,

                    directory=INPUT_IMAGES_DIR,

                    x_col="image",

                    y_col="season",

                    batch_size=32,

                    seed=42,

                    shuffle=True,

                    class_mode="categorical",

                    classes=["Spring", "Winter", "Fall", "Summer"],

                    target_size=(HEIGHT,WIDTH))



test_generator=test_datagen.flow_from_dataframe(

                    dataframe=test_df,

                    directory=INPUT_IMAGES_DIR,

                    x_col="image",

                    batch_size=1,

                    seed=42,

                    shuffle=False,

                    class_mode=None,

                    target_size=(HEIGHT,WIDTH))
from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D, Dropout

from keras import backend as K



from keras.applications.inception_v3 import InceptionV3

from keras.layers import Input



# this could also be the output a different Keras model or layer

input_tensor = Input(shape=(HEIGHT, WIDTH, 3))  # this assumes K.image_data_format() == 'channels_last'



# create the base pre-trained model

base_model = InceptionV3(input_tensor=input_tensor,weights='imagenet', include_top=False)



# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)

x = Dropout(0.25)(x)

x = Dense(128, activation='relu')(x)

x = Dropout(0.25)(x)

# and a logistic layer -- let's say we have 200 classes

predictions = Dense(4, activation='softmax')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional InceptionV3 layers

for layer in base_model.layers:

    layer.trainable = False



# compile the model (should be done *after* setting layers to non-trainable)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



# train the model on the new data for a few epochs

# classes=["Spring", "Winter", "Fall", "Summer"],



model.fit_generator(

        train_generator,

#         steps_per_epoch=2000,

        epochs=10,

        class_weight={0: 0.94, 1: 0.80, 2: 0.74, 3: 0.53},

        validation_data=valid_generator)

#         validation_steps=800)
from keras.models import load_model



model.save('/kaggle/input/my_model.h5')