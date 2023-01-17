import shutil

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from shutil import copyfile
%env JOBLIB_TEMP_FOLDER=/tmp
#shutil.rmtree('/tmp')
## Creating Directories



categories = list(np.arange(42))

categories = [str(i) for i in categories]

categories[0:10] = ['0' + i for i in categories[0:10]]



try: 

    

    base_dir = '/tmp/products/'

    os.mkdir(base_dir)

    

    train_dir = os.path.join(base_dir, 'training')

    validation_dir = os.path.join(base_dir, 'validation')

    os.mkdir(train_dir)

    os.mkdir(validation_dir)

    

    for cat in categories:

        os.mkdir(os.path.join(train_dir, str(cat)))

        os.mkdir(os.path.join(validation_dir, str(cat)))

        print(cat+' created successfully in both train and validation')

    

except OSError:

    pass
len(os.listdir('/tmp/products/validation'))
# Splitting files to Train/Valid sets (training on smaller subset)

import random

def split_data(source, train, validation, split_size, subset):

    

    all_files = []

    for file_name in os.listdir(source):

        file_path = source + file_name

        

        if os.path.getsize(file_path)>0:

            all_files.append(file_name)

        else:

            print('{} is zero length, so ignoring'.format(file_name))

    

    

    n_files = len(all_files)

    subset_size = int(n_files * subset)  ## 30% of all images

    split_point = int(subset_size * split_size)  ## split accordingly to the 30% images

    

    shuffled = random.sample(all_files, n_files)

    shuffled = shuffled[:subset_size]  ## getting 30% of shuffled images

    

    train_set = shuffled[:split_point] 

    test_set = shuffled[split_point:]

    

    for file_name in train_set:

        copyfile(source + file_name, train + file_name)

    

    for file_name in test_set:

        copyfile(source + file_name, validation + file_name)
import random

def specialised_data(source, train, validation):

    

    all_files = []

    for file_name in os.listdir(source):

        file_path = source + file_name

        

        if os.path.getsize(file_path)>0:

            all_files.append(file_name)

        else:

            print('{} is zero length, so ignoring'.format(file_name))

    

    

    n_files = len(all_files) 

    

    shuffled = random.sample(all_files, n_files)

    

    train_set = shuffled[:500] 

    test_set = shuffled[500:750]

    

    for file_name in train_set:

        copyfile(source + file_name, train + file_name)

    

    for file_name in test_set:

        copyfile(source + file_name, validation + file_name)
sources = ['/kaggle/input/shopee-dataset/train/train/' + cat + '/' for cat in categories]

training_source = ['/tmp/products/training/' + cat + '/' for cat in categories]

validation_source = ['/tmp/products/validation/' + cat + '/' for cat in categories]



# Train/Val Size

split_size = 0.95



# Set how much of the data you want to train with

subset = 1.0
for source, train, valid in zip(sources, training_source, validation_source):

    

    split_data(source, train, valid, split_size, subset)
print(len(os.listdir('/tmp/products/training/00')))

print(len(os.listdir('/tmp/products/validation/00')))
!pip install -U efficientnet
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers

from tensorflow.keras import Model

#from tensorflow.keras.applications.EfficientNetB7 import EfficientNetB7

from efficientnet.tfkeras import EfficientNetB7



img_size = 75



pre_trained_model = EfficientNetB7(input_shape=(img_size, img_size, 3),

                                      include_top=False,

                                      weights='imagenet')



#for layer in pre_trained_model.layers[0:700]:

    #layers.trainable = False

    

#for layer in pre_trained_model.layers[700:]:

    #layers.trainable = True

  

#pre_trained_model.summary()

#last_layer = pre_trained_model.get_layer('block8_7_mixed')

#last_output = last_layer.output



last_output = pre_trained_model.output
cost = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

optimize = tf.keras.optimizers.Adam(lr=0.001)



x = layers.Flatten()(last_output)

x = layers.Dense(64, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(0.05))(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(42, activation='softmax')(x)



model = Model(pre_trained_model.input, x)



model.compile(optimizer=optimize,

              loss=cost,

              metrics=['accuracy'])
model.summary()
train_dir = '/tmp/products/training'

validation_dir = '/tmp/products/validation'



train_datagen = ImageDataGenerator(rescale=1/255)





train_generator = train_datagen.flow_from_directory(train_dir,

                                                    batch_size=64,

                                                    class_mode='categorical',

                                                    target_size=(img_size,img_size),

                                                    color_mode = 'rgb')



validation_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = validation_datagen.flow_from_directory(validation_dir,

                                                              batch_size=32,

                                                              class_mode='categorical',

                                                              target_size=(img_size,img_size),

                                                              color_mode = 'rgb')
checkpoint_path = '/kaggle/working/EfficientNetB7.ckpt'

checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,

                                                 save_weights_only=True,

                                                 verbose=1)
#model.load_weights('/kaggle/input/gpuweights/InceptionResNetV2.ckpt')

model.load_weights('/kaggle/working/EfficientNetB7.ckpt')
history = model.fit(train_generator,

                    epochs = 2, 

                    verbose = 1,

                    validation_data = validation_generator,

                    callbacks=[cp_callback])
from tqdm import tqdm

import cv2



df_test = pd.read_csv('/kaggle/input/shopee-dataset/test.csv')

X_test = []



for imageName in tqdm(df_test['filename']): 

    image = cv2.imread('/kaggle/input/shopee-dataset/test/test/' + imageName)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (img_size,img_size))

    X_test.append(image)

    

X_test = np.array(X_test).astype('float32')/255
predictions = model.predict(X_test, batch_size=32)

preds = predictions.argmax(axis=-1)
preds
your_choice = 85

print('Your choice belongs to category {}'.format(preds[your_choice]))



from keras.preprocessing import image

import matplotlib.image as mpimg



file_names = os.listdir('/kaggle/input/shopee/test/test')

path = '/kaggle/input/shopee/test/test/' + file_names[your_choice]



test_img = mpimg.imread(path)

plt.figure(figsize=(8,8))

plt.imshow(test_img)

plt.show()
## Category Class

cat = '02'

path_cat = '/tmp/products/training/' + str(cat) + '/' + os.listdir('/tmp/products/training/'+ cat)[100]



plt.figure(figsize=(8,8))

cat_img = mpimg.imread(path_cat)

plt.imshow(cat_img)

plt.show()
df_test['category'] = preds

df_test['category'] = df_test.category.apply(lambda c : str(c).zfill(2))
df_test
df_test.to_csv('/kaggle/working/final.csv', index=False)