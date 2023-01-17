# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.chdir('/kaggle/input/new-plant-diseases-dataset')

os.getcwd()
check_dir = os.listdir()

print(check_dir)
default_dir = os.getcwd()
os.chdir(default_dir + '/' + 'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train')
current_dir = os.getcwd()
import tensorflow as tf



img_height = 224

img_width = 224

batch_size = 100



train_ds = tf.keras.preprocessing.image_dataset_from_directory(

  current_dir,

  validation_split=0.2,

  subset="training",

  seed=123,

  image_size=(img_height, img_width),

  batch_size=batch_size)

train_ds.class_names
work_class = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',

              'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Pepper,_bell___Bacterial_spot', 

              ',Pepper,_bell___healthy','Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 

              'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 

              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 

              'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',

              'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 

              'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 

              'Potato___healthy	Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 

              'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',

              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 

              'Tomato___healthy'

]



len(work_class)