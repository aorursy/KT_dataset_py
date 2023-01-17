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
import os
import pandas as pd
import numpy as np
import cv2
import random
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

os.mkdir('generated-nevusNonnevus')
folder = ['train','val']
sub_folder = ['nevus','nonnevus']
for i in folder:
    os.mkdir(f'generated-nevusNonnevus/{i}')
    for x in sub_folder:
        os.mkdir(f'generated-nevusNonnevus/{i}/{x}')
data= []
Ydata = []
height = 320
width = 320
# total train images per  class 4890
import cv2
count = 0
length = len(os.listdir("generated-nevusNonnevus/train/nevus"))
print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/derma-diseases/dataset/train/nevus')):
    x = cv2.imread(f'../input/derma-diseases/dataset/train/nevus/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(0)

# total train images per  class 4895
import cv2
length = len(os.listdir("generated-nevusNonnevus/val/nevus"))
print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/derma-diseases/dataset/validation/nevus')):
    x = cv2.imread(f'../input/derma-diseases/dataset/validation/nevus/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(0)
# total train images per  class 4895
import cv2
length = len(os.listdir("generated-nevusNonnevus/val/nevus"))
print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/derma-diseases/dataset/test/nevus')):
    x = cv2.imread(f'../input/derma-diseases/dataset/test/nevus/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(0)
len(os.listdir("generated-nevusNonnevus/train/nevus"))
# total train images per  class 4895
import cv2
length = len(os.listdir("generated-nevusNonnevus/val/nevus"))
print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/train/nevus')):
    x = cv2.imread(f'../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/train/nevus/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(0)

print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/test/nevus')):
    x = cv2.imread(f'../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/test/nevus/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(0)
print(len(os.listdir("generated-nevusNonnevus/train/nevus")))
print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/valid/nevus')):
    x = cv2.imread(f'../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/valid/nevus/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(0)
length = len(os.listdir("generated-nevusNonnevus/val/nevus"))
print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/dermoscopy-images/train/train/nv')):
    x = cv2.imread(f'../input/dermoscopy-images/train/train/nv/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(0)
length = len(os.listdir("generated-nevusNonnevus/val/nevus"))
print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/dermoscopy-images/val/val/nv')):
    x = cv2.imread(f'../input/dermoscopy-images/val/val/nv/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(0)
#val images 1008
print(len(os.listdir("generated-nevusNonnevus/train/nevus")))
count = 0
print("Getting from Non-Dataset1")
for img in  tqdm_notebook(os.listdir('../input/derma-diseases/dataset/train/melanoma')):
    x = cv2.imread(f'../input/derma-diseases/dataset/train/melanoma/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)
for img in  tqdm_notebook(os.listdir('../input/derma-diseases/dataset/train/seborrheic_keratosis')):
    x = cv2.imread(f'../input/derma-diseases/dataset/train/seborrheic_keratosis/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)
print(f"===> total nonnevus{len(os.listdir('generated-nevusNonnevus/train/nonnevus/'))}")
print("Getting from Non-Dataset1")
for img in  tqdm_notebook(os.listdir('../input/derma-diseases/dataset/test/melanoma')):
    x = cv2.imread(f'../input/derma-diseases/dataset/test/melanoma/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)
for img in  tqdm_notebook(os.listdir('../input/derma-diseases/dataset/test/seborrheic_keratosis')):
    x = cv2.imread(f'../input/derma-diseases/dataset/test/seborrheic_keratosis/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)
print(f"===> total nonnevus{len(os.listdir('generated-nevusNonnevus/train/nonnevus/'))}")
print("Getting from Non-Dataset1")
for img in  tqdm_notebook(os.listdir('../input/derma-diseases/dataset/validation/melanoma')):
    x = cv2.imread(f'../input/derma-diseases/dataset/validation/melanoma/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)
for img in  tqdm_notebook(os.listdir('../input/derma-diseases/dataset/validation/seborrheic_keratosis')):
    x = cv2.imread(f'../input/derma-diseases/dataset/validation/seborrheic_keratosis/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)
print(f"===> total nonnevus{len(os.listdir('generated-nevusNonnevus/train/nonnevus/'))}")

print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/dermoscopy-images/train/train/les')):
    x = cv2.imread(f'../input/dermoscopy-images/train/train/les/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)
print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/dermoscopy-images/val/val/les')):
    x = cv2.imread(f'../input/dermoscopy-images/val/val/les/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)
print(f"===> total nonnevus:  {len(os.listdir('generated-nevusNonnevus/train/nonnevus/'))}")

print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/train/melanoma')):
    x = cv2.imread(f'../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/train/melanoma/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)

print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/test/melanoma')):
    x = cv2.imread(f'../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/test/melanoma/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)
print(f"===> total nonnevus:  {len(os.listdir('generated-nevusNonnevus/train/nonnevus/'))}")
print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/train/seborrheic_keratosis')):
    x = cv2.imread(f'../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/train/seborrheic_keratosis/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)

print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/test/seborrheic_keratosis')):
    x = cv2.imread(f'../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/test/seborrheic_keratosis/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)
print(f"===> total nonnevus:  {len(os.listdir('generated-nevusNonnevus/train/nonnevus/'))}")

print("Getting from Dataset1")
for img in  tqdm_notebook(os.listdir('../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/valid/melanoma')):
    x = cv2.imread(f'../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/valid/melanoma/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)
for img in  tqdm_notebook(os.listdir('../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/valid/seborrheic_keratosis')):
    x = cv2.imread(f'../input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/valid/seborrheic_keratosis/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)
print(f"===> total nonnevus:  {len(os.listdir('generated-nevusNonnevus/train/nonnevus/'))}")
for img in  tqdm_notebook(os.listdir('../input/skin-cancer-malignant-vs-benign/test/benign')):
    x = cv2.imread(f'../input/skin-cancer-malignant-vs-benign/test/benign/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)
print(f"===> total nonnevus:  {len(os.listdir('generated-nevusNonnevus/train/nonnevus/'))}")
for img in  tqdm_notebook(os.listdir('../input/skin-cancer-malignant-vs-benign/train/benign')):
    x = cv2.imread(f'../input/skin-cancer-malignant-vs-benign/train/benign/{img}')
    #print(x)
    data.append(cv2.resize(x,(height,width)))
    Ydata.append(1)

print(f"===> total nonnevus:  {len(os.listdir('generated-nevusNonnevus/train/nonnevus/'))}")
len(os.listdir('generated-nevusNonnevus/train/nonnevus/')),len(os.listdir('generated-nevusNonnevus/train/nevus/'))
6267 - 5394
for i in os.listdir('generated-nevusNonnevus/train/nevus/')[5394:]:
    os.remove(f'generated-nevusNonnevus/train/nevus/{i}')
for i in os.listdir('generated-nevusNonnevus/train/nonnevus/')[5394:]:
    os.remove(f'generated-nevusNonnevus/train/nonnevus/{i}')
len(os.listdir('generated-ne}vusNonnevus/train/nonnevus/')),len(os.listdir('generated-nevusNonnevus/train/nevus/'))

for i in os.listdir('generated-nevusNonnevus/train/nevus/')[4890:]:
    x = cv2.imread(f'generated-nevusNonnevus/train/nevus/{i}')
    cv2.imwrite(f'generated-nevusNonnevus/val/nevus/{i}',x)
for i in os.listdir('generated-nevusNonnevus/train/nonnevus/')[4890:]:
    print(i)
    x = cv2.imread(f'generated-nevusNonnevus/train/nonnevus/{i}')
    cv2.imwrite(f'generated-nevusNonnevus/val/nonnevus/{i}',x)
len(os.listdir(f'generated-nevusNonnevus/train/nonnevus'))
cv2.imread(f'generated-nevusNonnevus/train/nonnevus/{i}')
len(os.listdir('generated-nevusNonnevus/val/nonnevus/')),len(os.listdir('generated-nevusNonnevus/val/nevus/'))
len(data), len(Ydata)
import keras 
from keras import models
from keras.applications import ResNet50
from keras.layers import Dense, Dropout, Flatten
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as  plt
from keras import models
from tensorflow.keras.models import load_model,model_from_json
conv_base = ResNet50(weights='imagenet',
include_top=False,
input_shape=(320,320 , 3))

print(conv_base.summary())
model = models.Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

print(model.summary())
# Make last block of the conv_base trainable:

for layer in conv_base.layers[:165]:
   layer.trainable = False
for layer in conv_base.layers[165:]:
   layer.trainable = True

print('Last block of the conv_base is now trainable')
for i, layer in enumerate(conv_base.layers):
   print(i, layer.name, layer.trainable)
model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("model compiled")
print(model.summary())
# Prep the Train Valid and Test directories for the generator
'''
train_dir = 'generated-nevusNonnevus/train/'
validation_dir = 'generated-nevusNonnevus/val'
batch_size = 20
target_size=(224, 224)

#train_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,target_size=target_size,batch_size=batch_size,shuffle = True)
validation_generator = val_datagen.flow_from_directory(
    validation_dir,target_size=target_size,batch_size=batch_size,shuffle = True)
'''

history = model.fit(np.array(data),np.array(Ydata),epochs =5,batch_size=1)
9780/2
