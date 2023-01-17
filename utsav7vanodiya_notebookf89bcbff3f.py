# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json



with open("../input/til2020/train.json",'r') as file:

    train_data = json.load(file)

    

with open("../input/til2020/val.json",'r') as file:

    test_data = json.load(file)

    

print("%.1000s" % train_data)
train_annotations = train_data['annotations']

train_images = train_data['images']

categories = train_data['categories']



print(train_annotations[0])

print(train_images[0])

print(categories)



print(len(train_annotations))
test_annotations = test_data['annotations']

test_images = test_data['images']



print(test_annotations[0])

print(test_images[0])



print(len(test_annotations))
category_mapping = {}



for category_item in categories:

    category_mapping[category_item['id']] = category_item['name']



train_id_to_path_mapping = {}



for image_item in train_images:

    train_id_to_path_mapping[image_item['id']] = image_item['file_name']

    

test_id_to_path_mapping = {}



for image_item in test_images:

    test_id_to_path_mapping[image_item['id']] = image_item['file_name']

    

for annotation in train_annotations:

    annotation['image_path'] = train_id_to_path_mapping[annotation['image_id']]

    annotation['cat'] = category_mapping[annotation['category_id']]

    annotation['bbox'] = list(map(int,annotation['bbox']))

    

for annotation in test_annotations:

    annotation['image_path'] = test_id_to_path_mapping[annotation['image_id']]

    annotation['cat'] = category_mapping[annotation['category_id']]

    annotation['bbox'] = list(map(int,annotation['bbox']))

    

print("%.1000s" % train_annotations)

print("%.1000s" % test_annotations)
from matplotlib import pyplot as plt

import cv2



base_path = r'../input/til2020/train/train/'



annotation = train_annotations[0]

coordinates = annotation['bbox']



im = cv2.imread(base_path+annotation['image_path'])

#Show the image with matplotlib

print(im.shape)

plt.imshow(im)

plt.show()



from PIL import Image, ImageFont, ImageDraw



img = Image.open(base_path+annotation['image_path'])

# create rectangle image

img1 = ImageDraw.Draw(img)

img1.rectangle(((coordinates[0], coordinates[1]),(coordinates[0]+coordinates[2], coordinates[1]+coordinates[3])),  outline ="red",width=10)

img1.text((523,25), "dresses", fill=(255,255,255,128))

display(img)

image = im[coordinates[1]:coordinates[1]+coordinates[3],coordinates[0]:coordinates[0]+coordinates[2]]



plt.imshow(image)

plt.show()

print(image.shape)
resized_image = cv2.resize(image, (128, 128),  

               interpolation = cv2.INTER_NEAREST) 



plt.imshow(resized_image)

plt.show()

print(resized_image.shape)
max_images = 10



for annotation in train_annotations[:max_images]:

    coordinates = annotation['bbox']

    

    x = coordinates[0]

    y = coordinates[1]

    width = coordinates[2]

    height = coordinates[3]



    im = cv2.imread(base_path+annotation['image_path'])

    image = im[y:y+height,x:x+width]



    resized_image = cv2.resize(image, (128, 128), interpolation = cv2.INTER_NEAREST) 

    #resized_image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_NEAREST) 

    #resized_image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_NEAREST) 

    plt.imshow(resized_image)

    plt.title(annotation['cat'])

    plt.show()

    #print(annotation['cat'])

from matplotlib import pyplot as plt

from PIL import Image, ImageFont, ImageDraw

    

import numpy as np

import cv2

import keras



#, original_shape, new_shape=(224,224)



def get_cropped_image(img, bbox):

    start_x, start_y, width, height = bbox

    cropped_img = img[start_y:start_y+height, start_x:start_x+width]

    return cropped_img

      

def get_reshaped_image(img, new_shape=(224,224)):

    resized_image = cv2.resize(img, new_shape, interpolation = cv2.INTER_NEAREST) 

    return resized_image



def rescale_bbox(bbox, current_img_shape, new_img_shape=(224,224)):

    x_ratio = new_img_shape[0] / current_img_shape[0]

    y_ratio = new_img_shape[1] / current_img_shape[1]

    

    new_x = bbox[0] * x_ratio

    new_y = bbox[1] * y_ratio

    new_width = bbox[2] * x_ratio

    new_height = bbox[3] * y_ratio

    

    return new_x, new_y, new_width, new_height

   
# Temporary Test

max_images = 1

title_distance = 10



for annotation in train_annotations[:max_images]:

    bbox = annotation['bbox']



    img = Image.open(base_path+annotation['image_path'])

    img2 = np.asarray(image)

    

    x,y,w,h = rescale_bbox(bbox, (img2.shape[0],img2.shape[1]))

    cropped_image = get_cropped_image(img2, bbox)

    reshaped_image = get_reshaped_image(cropped_image)



    img = Image.fromarray(reshaped_image)

    img3 = ImageDraw.Draw(img)

    img3.rectangle(((x, y),(x+w, y+h)),  outline ="green",width=1)

    #img3.text((x-title_distance,y-title_distance), "dresses", fill=(255,255,255,128))

    display(img)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style()



# to divide our data into train and validation set

from sklearn.model_selection import train_test_split

#to encode our labels

from tensorflow.keras.utils import to_categorical

#to build our model 

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout

# Our optimizer options

from keras.optimizers import RMSprop

from keras.optimizers import Adam

#Callback options

from tensorflow.keras.callbacks import EarlyStopping

from keras.callbacks import ReduceLROnPlateau

#importing image data generator for data augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#for the final prediction report

from sklearn.metrics import classification_report ,confusion_matrix
new_categories = [x['name'] for x in categories]

print(new_categories)

encoded_categories = to_categorical(list(range(len(new_categories))), num_classes=len(new_categories))

print(encoded_categories)



category_mapping = {x:encoded_categories[i] for i,x in enumerate(new_categories)}

print(category_mapping)
import cv2

import traceback

import sys



# dresses_count = 0

# trousers_count = 0



def transform_data(annotations, base_dir, samples_per_cat=None, cats=None):

    features = []

    labels = []

    max_check = False

    cat_count = {}

    

    if samples_per_cat is not None:

        max_check = True

        cat_count = {x:0 for x in cats}

    else:

        samples_per_cat = sys.maxsize

        

    

    for i, annotation in enumerate(annotations):

        img_path = annotation['image_path']

        cat = annotation['cat']

        bbox = annotation['bbox']



        try:

            if max_check:

                if cat in cats:

                    if cat_count[cat] >= samples_per_cat:

                        continue

                else:

                    continue



        #     if cat == 'trousers':

        #         trousers_count +=1



        #         if trousers_count > max_img:

        #             continue

        #     elif cat == 'dresses':

        #         dresses_count +=1



        #         if dresses_count > max_img:

        #             continue

        #     else:

        #         continue



            img = cv2.imread(base_dir+img_path)



            if img is None:

                continue

            

            #x,y,w,h = rescale_bbox(bbox, (img.shape[0],img.shape[1]))

            cropped_image = get_cropped_image(img, bbox)

            resized_image = get_reshaped_image(cropped_image, new_shape=(128,128))



            features.append(resized_image)

            labels.append(category_mapping[cat])



            cat_count[cat] += 1

            

            if i != 0 and i % 1000 == 0:

                print("Processed Images: ",i)



            #print(resized_image.shape)



            #plt.imshow(resized_image)

            #plt.title(cat)

            #plt.show()

        except:

            print(f"Error in image: bbox={bbox}, img_path={img_path}, cat={cat}")

            traceback.print_exc()

        

    return features, labels

    

    

max_samples = 10000

# cats = {'tops','trousers'}

cats = set(new_categories)

    

train_features, train_labels = transform_data(train_annotations, r'../input/til2020/train/train/',samples_per_cat=max_samples, cats=cats)

    

print(len(train_features))

print(len(train_labels))



#print(train_data[0])

#print(labels[0])



test_features, test_labels = transform_data(test_annotations, r'../input/til2020/val/val/',samples_per_cat=max_samples, cats=cats)

    

print(len(test_features))

print(len(test_labels))

    
print(len(train_features))

print(len(train_labels))



train_features_2 = np.asarray(train_features)

print(train_features_2.shape)

train_labels_2 = np.asarray(train_labels)

print(train_labels_2.shape)



print(len(test_features))

print(len(test_labels))



test_features_2 = np.asarray(test_features)

print(test_features_2.shape)

test_labels_2 = np.asarray(test_labels)

print(test_labels_2.shape)
from sklearn.model_selection import train_test_split



input_shape = (128, 128, 3)

epoch = 100



model = Sequential()



model.add(Conv2D(64, kernel_size=(5,5), input_shape=input_shape, activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, kernel_size=(3,3), input_shape=input_shape, activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape, activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape, activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(len(category_mapping), activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



# x_train, x_test, y_train, y_test = train_test_split(train_data2, labels2, test_size=0.2, random_state=1)



early_stop= EarlyStopping(monitor='val_loss',patience=10)



model.fit(train_features_2, train_labels_2,

          epochs=epoch,

          validation_data=(test_features_2,test_labels_2), 

          callbacks=[early_stop])
# example of loading the vgg16 model

from keras.applications.vgg16 import VGG16



vgg16 = VGG16(weights=None, input_shape=input_shape, classes=len(categories))



# summarize the model

vgg16.summary()



vgg16.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



early_stop= EarlyStopping(monitor='val_loss',patience=10)



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=10, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



vgg16.fit(train_features_2, train_labels_2,

          epochs=epoch,

          validation_data=(test_features_2,test_labels_2), 

          callbacks=[early_stop, learning_rate_reduction])



metrics=pd.DataFrame(vgg16.history.history)

metrics
# example of loading the vgg16 model

from keras.applications.resnet50 import ResNet50



resnet_50 = ResNet50(weights=None, input_shape=input_shape, classes=len(categories))



# summarize the model

resnet_50.summary()



resnet_50.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



early_stop= EarlyStopping(monitor='val_loss',patience=10)



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=10, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



resnet_50.fit(train_features_2, train_labels_2,

          epochs=epoch,

          validation_data=(test_features_2,test_labels_2), 

          callbacks=[early_stop, learning_rate_reduction])



metrics=pd.DataFrame(resnet_50.history.history)

metrics
# example of loading the vgg16 model

from keras.applications.inception_v3 import InceptionV3



inception_v3 = InceptionV3(weights=None, input_shape=input_shape, classes=len(categories))



# summarize the model

inception_v3.summary()



inception_v3.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



early_stop= EarlyStopping(monitor='val_loss',patience=10)



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=10, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



inception_v3.fit(train_features_2, train_labels_2,

          epochs=epoch,

          validation_data=(test_features_2,test_labels_2), 

          callbacks=[early_stop, learning_rate_reduction])



metrics=pd.DataFrame(inception_v3.history.history)

metrics