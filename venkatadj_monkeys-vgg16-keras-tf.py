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
from tensorflow.python.keras.models import Model,model_from_json
from tensorflow.python.keras.optimizers import adam,SGD
from tensorflow.python.keras.layers import Dense,Flatten
from tensorflow.python.keras.applications.vgg16 import preprocess_input
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
np.random.seed(43)
def load_dataset(self):
    train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    valid_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    self.train_generator = train_data_generator.flow_from_directory(
        self.train_data,
        target_size=(self.img_height,self.img_width),
        batch_size=self.batch_size,
        seed=42,
        class_mode='categorical')
    self.valid_generator = valid_data_generator.flow_from_directory(
        self.valid_data,
        target_size=(self.img_height,self.img_width),
        batch_size=self.batch_size,
        seed=42,
        class_mode='categorical')
    self.test_generator = valid_data_generator.flow_from_directory(
        self.valid_data,
        target_size=(self.img_height,self.img_width),
        batch_size=self.batch_size,
        seed=42,
        shuffle=False,
        class_mode=None)
def load_model(self):
    base_model=VGG16(weights='imagenet',input_shape=(self.img_height,self.img_width, 3),include_top=False)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(self.no_of_classes, activation='softmax')(x)
    self.model = Model(base_model.inputs,outputs=x)
    for layer in base_model.layers:
        layer.trainable = False
def build_model(self):
    self.model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    train_steps=int(self.train_generator.samples/self.batch_size)+1
    valid_steps=int(self.valid_generator.samples/self.batch_size)+1
    #from pdb import set_trace as bp
    #bp()
    self.model.fit_generator(self.train_generator,
          steps_per_epoch=train_steps,
          epochs=self.no_of_epochs,
          validation_data=self.valid_generator,
          validation_steps=valid_steps)

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
    self.model=None
    self.model = model_from_json(loaded_model_json)
    self.model.load_weights("%s.h5" % self.save_model_name)
    print("Loaded model from disk")
def predict_monkeys(self):
    valid_steps=int(self.valid_generator.samples/self.batch_size)+1
    predict_model = self.model.predict_generator(generator=self.test_generator, steps=valid_steps)
    for row in range(0, 3):
        plt.title("label=%s" % self.class_labels[str(predict_model.argmax(axis=1)[row])])
        img = mpimg.imread(self.valid_data+'/'+self.test_generator.filenames[row])
        imgplot = plt.imshow(img)
        plt.show()
class classify_monkeys:
    def __init__(self):
        self.train_data=os.path.expanduser('/kaggle/input/10-monkey-species/training/training')
        self.valid_data=os.path.expanduser('/kaggle/input/10-monkey-species/validation/validation')
        self.class_labels={'0':'alouattapalliata','1':'erythrocebuspatas','2':'cacajaocalvus','3':'macacafuscata','4':'cebuellapygmea','5':'cebuscapucinus','6':'micoargentatus','7':'saimirisciureus','8':'aotusnigriceps','9':'trachypithecusjohnii'}
        self.batch_size=64
        self.img_height=224
        self.img_width=224
        self.no_of_epochs=3
        self.train_generator=None
        self.no_of_classes=10
        self.save_model_name='kaggle_monkeys'
        self.model=None
        self.new_model=None
    load_dataset=load_dataset
    load_model=load_model     
    build_model=build_model
    save_model=save_model
    load_saved_model=load_saved_model
    save_model=save_model
    predict_monkeys=predict_monkeys
def monkeys_from_scratch():
    classify=classify_monkeys()
    classify.load_dataset()
    classify.load_model()
    classify.build_model()
    classify.save_model()
    classify.load_saved_model()
    classify.predict_monkeys()


def monkeys_reuse_saved_model():
    classify=classify_monkeys()
    classify.load_dataset()
    classify.load_saved_model()
    classify.predict_monkeys()

def main():
    monkeys_from_scratch()
    #monkeys_reuse_saved_model()
if __name__=='__main__':
    main()
