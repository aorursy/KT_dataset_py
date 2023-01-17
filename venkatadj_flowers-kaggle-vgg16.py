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
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import adam,SGD
from tensorflow.python.keras.layers import Dense,Flatten
from tensorflow.python.keras.applications.vgg16 import preprocess_input
import os
import numpy as np
np.random.seed(43)
def load_dataset(self):
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    self.train_generator = data_generator.flow_from_directory(
        self.test_data,
        target_size=(320,240),
        batch_size=self.batch_size,
        seed=42,
        class_mode='categorical')

def load_model(self):
    base_model=VGG16(weights='imagenet',input_shape=(320, 240, 3),include_top=False)
    x = base_model.output
    x = Flatten()(x)
    #x = Dense(1024, activation='softmax')(x)
    x = Dense(self.no_of_classes, activation='softmax')(x)
    self.model = Model(base_model.inputs,outputs=x)
    for layer in base_model.layers:
        layer.trainable = False
def build_model(self):
    self.model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    steps=int(self.train_generator.samples/self.batch_size)+1
    self.model.fit_generator(self.train_generator,
          steps_per_epoch=steps,
          epochs=self.no_of_epochs)

def save_model(self):
    model_json = self.model.to_json()
    with open("%s.json" % self.save_model_name, "w") as json_file:
        json_file.write(model_json)
    self.model.save_weights("%s.h5" % self.save_model_name)

def load_saved_model(self):
    print("Loading model from disk")
    json_file = open('%s.json' % self.save_model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.new_model = model_from_json(loaded_model_json)
    self.new_model.load_weights("%s.h5" % self.save_model_name)
    print("Loaded model from disk")

class classify_flowers:
    def __init__(self):
        self.test_data=os.path.expanduser('/kaggle/working/flowers')
        self.image_size=224
        self.batch_size=64
        self.train_generator=None
        self.no_of_classes=5
        self.save_model_name='kaggle_flowers'
        self.model=None
        self.new_model=None
        self.no_of_epochs=2
    load_dataset=load_dataset
    load_model=load_model
    build_model=build_model
    save_model=save_model
    load_saved_model=load_saved_model
    save_model=save_model

def main():
    classify=classify_flowers()
    classify.load_dataset()
    classify.load_model()
    classify.build_model()
    classify.save_model()
    #classify.load_model()

import subprocess as sub
sub.check_output('rm -rf /kaggle/working/flowers',shell=True)
sub.check_output('cp -rf /kaggle/input/flowers-recognition/flowers/flowers /kaggle/working',shell=True)
sub.check_output('ls /kaggle/working/flowers',shell=True)
main()
