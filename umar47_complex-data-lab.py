import warnings

warnings.filterwarnings("ignore")

import numpy as np # linear algebra

import matplotlib.pyplot as plt

import os

import json

import cv2

import pandas as pd

from pandas.io.json import json_normalize

from collections import Iterable



pd.set_option('display.max_rows', 5000)

print(os.listdir("../input")),

# Any results you write to the current directory are saved as output.
df_train=pd.read_json('../input/solesensei_bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json')

df_val=pd.read_json('../input/solesensei_bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json')
#My solution to get a dict inside json found here:https://stackoverflow.com/questions/48942801/json-to-csv-output-using-pandas

from keras.utils import to_categorical

json_list = [j[1][1] for j in df_train.iterrows()]

result = pd.DataFrame(json_list)

result.to_csv("my_csv_file.csv", index=False)

train_attribute=pd.read_csv('my_csv_file.csv')

train_attribute=pd.get_dummies(train_attribute)#turn data to categorical like format for image classification

train_attribute=to_categorical(train_attribute, 9)

print(train_attribute)
json_list = [j[1][1] for j in df_train.iterrows()]

result_val = pd.DataFrame(json_list)

result_val.to_csv("att_train.csv", index=False)

train_attribute=pd.read_csv('att_train.csv')

print(train_attribute.head())
image_names = [j[1][0] for j in df_train.iterrows()]

result_train = pd.DataFrame(image_names)

result_train.to_csv("names.csv", index=False)

names_train=pd.read_csv('names.csv')

names_train.rename(columns={'0': 'name'}, inplace=True)

print(names_train.head())#labels_val.head())
json_list_category = [j[1][3][3] for j in df_val.iterrows()]

result_val = pd.DataFrame(json_list_category)

result_val.to_csv("labels.csv", index=False)

labels_val=pd.read_csv('labels.csv')

print(labels_val.head())
json_list_category = [j[1][3][3]['category'] for j in df_train.iterrows()]

result_train = pd.DataFrame(json_list_category)

result_train.to_csv("category.csv", index=False)

category_train=pd.read_csv('category.csv')

category_train.rename(columns={'0':'category'}, inplace=True)

print(category_train.head())
json_list_category = [j[1][3][3]['attributes'] for j in df_val.iterrows()]

result_val = pd.DataFrame(json_list_category)

result_val.to_csv("attributes.csv", index=False)

attributes_val=pd.read_csv('attributes.csv')

print(attributes_val)
df_train_concated=pd.concat([names_train, category_train, train_attribute], axis=1)

print(df_train_concated.head())
json_list_category = [j[1][3][3]['manualShape'] for j in df_val.iterrows()]

result_val = pd.DataFrame(json_list_category)

result_val.to_csv("manualShape_val.csv", index=False)

manualShape_val=pd.read_csv('manualShape_val.csv')

print(manualShape_val.head())
json_list_category = [j[1][3][3]['manualAttributes'] for j in df_val.iterrows()]

result_val = pd.DataFrame(json_list_category)

result_val.to_csv("manualAttribute_val.csv", index=False)

manualAttribute_val=pd.read_csv('manualAttribute_val.csv')

print(manualAttribute_val.head())
ctgry=[j[1][3] for j in df_train.iterrows()]

print(ctgry[0])
json_list_category = [j[1][3][0] for j in df_val.iterrows()]

result_val = pd.DataFrame(json_list_category)

result_val.to_csv("box2d_val.csv", index=False)

manualAttribute_val=pd.read_csv('box2d_val.csv')

print(manualAttribute_val[0])
print("total train labels: ", len(df_train['labels']))
path='../input/solesensei_bdd100k/bdd100k/bdd100k/images/100k'

train = sorted([os.path.join(path, 'train', file)

         for file in os.listdir(path + "/train")])

Val = sorted([os.path.join(path, 'val', file)

         for file in os.listdir(path + "/val")])

testA = sorted([os.path.join(path, 'train/testA', file)

         for file in os.listdir(path + "/train/testA")])

testB = sorted([os.path.join(path, 'train/testB', file)

         for file in os.listdir(path + "/train/testB")])   

trainA = sorted([os.path.join(path, 'train/trainA', file)

         for file in os.listdir(path + "/train/trainA")])

trainB = sorted([os.path.join(path, 'train/trainB', file)

         for file in os.listdir(path + "/train/trainB")])        

Train=testA + testB + trainA + trainB + train 



test_image=Val[0]#np.random.choice(Train)

tets=cv2.imread(test_image)#, cv2.IMREAD_GRAYSCALE)

plt.imshow(tets)

print(Val[0])

#df_train_concated['name']=='b1c66a42-6f7d68ca.jpg')

print(len(Train), len(Val))

def load_and_show_img_train(path):#COLOR LABELS TRAİN IMAGES

    train_data=sorted([os.path.join(path, 'train', file)

                for file in os.listdir(path + '/train')])

     

    show_test_image=train_data[0]#np.random.choice(train_data)

    show=cv2.imread(show_test_image, cv2.IMREAD_COLOR)

    plt.imshow(show)

    print(show_test_image)

path='../input/solesensei_bdd100k/bdd100k_seg/bdd100k/seg/color_labels'

load_and_show_img_train(path)
def load_and_show_img_val(path):#color_labels path,this is segmented images

    validation_data=sorted([os.path.join(path, 'val', file)

                for file in os.listdir(path + '/val')])

     

    show_test_image=validation_data[0]

    show=cv2.imread(show_test_image, cv2.IMREAD_COLOR)

    plt.imshow(show)

    

path='../input/solesensei_bdd100k/bdd100k_seg/bdd100k/seg/color_labels'

load_and_show_img_val(path)
path='../input/solesensei_bdd100k/bdd100k_seg/bdd100k/seg/labels'

load_and_show_img_train(path)



path='../input/solesensei_bdd100k/bdd100k_seg/bdd100k/seg/labels'

load_and_show_img_val(path)
path='../input/solesensei_bdd100k/bdd100k_seg/bdd100k/seg/images'

train = sorted([os.path.join(path, 'train', file)

         for file in os.listdir(path + "/train")])

test = sorted([os.path.join(path, 'test', file)

         for file in os.listdir(path + "/test")])

val = sorted([os.path.join(path, 'val', file)

         for file in os.listdir(path + "/val")])

print(len(train), len(test), len(val))



test_image=np.random.choice(test)

tets=cv2.imread(test_image)#, cv2.IMREAD_GRAYSCALE)

plt.imshow(tets)

    