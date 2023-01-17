from matplotlib import pyplot as plt

import cv2

from tqdm import tnrange, tqdm_notebook

from time import sleep
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from tensorflow.keras.applications import densenet

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

import os



x_files = list()

y_files = None



for dirname, _, filenames in os.walk('/kaggle/input'):

    folder_name = dirname.split(os.sep)[-1]

    if(folder_name != "test"):

        for filename in filenames:

            if(filename.endswith(".jpg")):

                x_files.append(os.path.join(dirname, filename))

            else:

                if(filename == "train.csv"):

                    y_files = os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
y_files
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,

                                  validation_split=0.5)

train_generator = train_datagen.flow_from_directory('/kaggle/input/shopee-code-league-2020-product-detection/resized/train', 

                                                    target_size=(224,224), 

                                                    color_mode='rgb', 

                                                    batch_size=32, 

                                                    class_mode='categorical', 

                                                    shuffle=True,

                                                    subset='training')

category_dict = train_generator.class_indices

print(category_dict)
number_of_classes = len(category_dict)

base_model = densenet.DenseNet121(weights='imagenet', include_top=False)

#densenet.DenseNet121(weights=’imagenet',include_top=False)



x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu')(x)

x = Dense(512, activation='relu')(x)

x = Dense(256, activation='relu')(x)



preds = Dense(number_of_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)



# Print the updated layer names.

# for i,layer in enumerate(model.layers): print(i,layer.name)

# Set the first n_freeze layers of the network to be non-trainable.

n_freeze = 300



for layer in model.layers[:n_freeze]:

    layer.trainable=False

for layer in model.layers[n_freeze:]:

    layer.trainable=True
for layer in model.layers:

    layer.trainable=False
from tensorflow.keras.optimizers import SGD

opt = SGD(lr=0.5)

model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy'])

step_size_train = train_generator.n//train_generator.batch_size

model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=5)
# Without transfer learning.

default_model = densenet.DenseNet121(weights='imagenet')

#densenet.DenseNet121(weights=’imagenet',include_top=True)
print(x)
prediction = model.evaluate(x)

keras.applications.densenet.decode_predictions(prediction, top=2)
test_path = '/kaggle/input/shopee-code-league-2020-product-detection/resized/test'

for directory in os.listdir(test_path):

 

    img_path = test_path+directory

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    

    preds = model.predict(x)

    default_preds = default_model.predict(x)

    

    # Printing results.

    

    # Default 1000 classes (without transfer learning).

    print(f"Without Transfer Learning Top-2 [{directory}]: \n{decode_predictions(default_preds, top=2)[0]}\n")



    # Print transfer learning model top-1

    confidence_array = preds[0]

    index_max = np.argmax(confidence_array)

    

    # Get KEY (category) by VALUE (index_max) in dictionary

    # mydict = {‘george’:16,’amber’:19}

    # print(list(mydict.keys())[list(mydict.values()).index(16)]) # Example in one line.



    category_names = category_dict.keys()

    category_values = category_dict.values()

    category_at_index = list(category_values).index(index_max)

    category_max = list(category_names)[category_at_index]

    

    print(f"\nWith Transfer Learning [{directory}]: \nTop-1 (confidence)\n{category_max} ({max(confidence_array)*100}%)")

    

    # Print transfer learning model all classes

    print("\nClass (confidence)")



    for category in category_dict:

        category_index = category_dict[category]

        value = confidence_array[category_index] * 100

        print(f"{category} ({value}%)")



    print("\n============================\n")