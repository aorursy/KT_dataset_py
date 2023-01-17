# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# get file name

TRAIN_NORMAL = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/"

TRAIN_PNEUMONIA = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/"

TEST_NORMAL = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL/"

TEST_PNEUMONIA = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/"

VAL_NORMAL = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/NORMAL/"

VAL_PNEUMONIA = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/PNEUMONIA/"



folder_list = [TRAIN_NORMAL, TEST_NORMAL, VAL_NORMAL, TRAIN_PNEUMONIA, TEST_PNEUMONIA, VAL_PNEUMONIA]

ext_list = ['jpeg', 'jpg', 'bmp', 'gif']

data_list = []

label_list = []

sel_label_list = []



sel_label_set = ["train", "test", "val", "train", "test", "val"]



for i, folder in enumerate(folder_list):

    files = os.listdir(folder)

    file_list = []

    tmp_list = []



    label = 'PNUEMONIA' if i >= 3 else 'NORMAL'

    sel_label = sel_label_set[i]

    

    for file in files:

        if file.split('.')[-1] in ext_list:

            data_list.append(file)

            label_list.append(label)

            sel_label_list.append(sel_label)





df_total = pd.DataFrame({'data': data_list, 'label': label_list, 'sel_label': sel_label_list})

df_total.head(5)
df_total.tail(5)
df_total.groupby(['sel_label', 'label']).count()

import cv2

import matplotlib.pyplot as plt

import random



random.seed(0)



def sample_disp(df, sample_num=5, sel="train", label='NORMAL'):

    index_num = sample_num//5 if sample_num%5==0 else sample_num//5 + 1

    fig, axes = plt.subplots(index_num, 5, figsize=(32, 8*index_num))

    df_sample = df_total.query("sel_label== @sel & label== @label")

    data_array_idx = random.sample(range(df_sample.shape[0]), sample_num)

    label_offset = 0 if label=='NORMAL' else 1

    if sel == 'train':

        sel_offset = 0

    elif sel == 'test':

        sel_offset = 1

    else:

        sel_offset = 2

    dir_sel_idx = label_offset * 3 + sel_offset

    axes.flatten()

    for i in range(sample_num):

        img = cv2.imread(os.path.join(folder_list[dir_sel_idx], df_sample.iloc[data_array_idx[i], 0]))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[i].imshow(img)

        axes[i].set_title(df_sample.iloc[data_array_idx[i], 0])

    plt.show()
sample_disp(df_total, 5, "train", "NORMAL")
sample_disp(df_total, 5, "test", "PNUEMONIA")

sample_disp(df_total, 5, "val", "NORMAL")
sample_disp(df_total, 5, "test", "NORMAL")
def offset_calc(label, sel_label):

    label_offset = 0 if label == 'NORMAL' else 1

    

    if sel_label == 'train':

        sel_label_offset = 0

    elif sel_label == 'test':

        sel_label_offset = 1

    else:

        sel_label_offset = 2

        

    return label_offset * 3 + sel_label_offset
min_size_x = 5000

min_size_y = 5000



for i in range(df_total.shape[0]):

    dir_sel = folder_list[offset_calc(df_total['label'][i], df_total['sel_label'][i])]



    img = cv2.imread(os.path.join(dir_sel, df_total['data'][i]))

    min_size_y = min(min_size_y, img.shape[0])

    min_size_x = min(min_size_x, img.shape[1])

    

#print(min_size_y, min_size_x)
def image_resize(df, label, sel_label, shape=(64, 64)):

    img_list = []

    dir_sel = folder_list[offset_calc(label, sel_label)]

    #print(dir_sel)

    #print(df)

    

    for idx in df.index:

        #print(df['data'][idx])

        img = cv2.imread(os.path.join(dir_sel, df['data'][idx]))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (64, 64))

        img_list.append(img)

    

    return np.array(img_list)

train_normal_img_array = image_resize(df_total.query("label == 'NORMAL' & sel_label == 'train'"), "NORMAL", "train")

test_normal_img_array = image_resize(df_total.query("label == 'NORMAL' & sel_label == 'test'"), "NORMAL", "test")

val_normal_img_array = image_resize(df_total.query("label == 'NORMAL' & sel_label == 'val'"), "NORMAL", "val")



train_pnuemonia_img_array = image_resize(df_total.query("label == 'PNUEMONIA' & sel_label == 'train'"), "PNUEMONIA", "train")

test_pnuemonia_img_array = image_resize(df_total.query("label == 'PNUEMONIA' & sel_label == 'test'"), "PNUEMONIA", "test")

val_pnuemonia_img_array = image_resize(df_total.query("label == 'PNUEMONIA' & sel_label == 'val'"), "PNUEMONIA", "val")

train_img_array = np.concatenate([train_normal_img_array, train_pnuemonia_img_array], axis=0)

val_img_array = np.concatenate([val_normal_img_array, val_pnuemonia_img_array], axis=0)

test_img_array = np.concatenate([test_normal_img_array, test_pnuemonia_img_array], axis=0)



train_label_array = np.concatenate([df_total.query("label == 'NORMAL' & sel_label == 'train'")['label'].values, 

                                    df_total.query("label == 'PNUEMONIA' & sel_label == 'train'")['label'].values],

                                    axis=0)



val_label_array = np.concatenate([df_total.query("label=='NORMAL' & sel_label=='val'")['label'].values,

                                  df_total.query("label=='PNUEMONIA' & sel_label=='val'")['label'].values],

                                  axis=0)



test_label_array = np.concatenate([df_total.query("label=='NORMAL' & sel_label=='test'")['label'].values,

                                   df_total.query("label=='PNUEMONIA' & sel_label=='test'")['label'].values],

                                   axis=0)

shuffle_idx = np.arange(train_label_array.shape[0])

np.random.shuffle(shuffle_idx)



train_img_array = train_img_array[shuffle_idx]

train_label_array = train_label_array[shuffle_idx]



shuffle_idx = np.arange(val_label_array.shape[0])

np.random.shuffle(shuffle_idx)



val_img_array = val_img_array[shuffle_idx]

val_label_array = val_label_array[shuffle_idx]



shuffle_idx = np.arange(test_label_array.shape[0])

np.random.shuffle(shuffle_idx)



test_img_array = test_img_array[shuffle_idx]

test_label_array = test_label_array[shuffle_idx]

df_label_train = pd.DataFrame()

df_label_train['tmp'] = train_label_array.flatten()

df_label_train = pd.get_dummies(df_label_train['tmp'])



df_label_test = pd.DataFrame()

df_label_test['tmp'] = test_label_array.flatten()

df_label_test = pd.get_dummies(df_label_test['tmp'])



df_label_val = pd.DataFrame()

df_label_val['tmp'] = val_label_array.flatten()

df_label_val = pd.get_dummies(df_label_val['tmp'])



import tensorflow as tf

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Flatten, Dropout, Input

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.datasets import cifar10

from tensorflow.keras import optimizers



input_tensor = Input(shape=(64,64,3))

vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)



top_model = Sequential()

top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))

top_model.add(Dense(256, activation='sigmoid'))

top_model.add(Dropout(0.5))

top_model.add(Dense(2, activation='softmax'))

model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))



for layer in model.layers[:16]:

    layer.trainable = False

    

#model.compile(loss='categorical_crossentropy',

model.compile(loss='binary_crossentropy',

#              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              optimizer=optimizers.Adam(lr=1e-4),

              metrics=['accuracy'])



#model.load_weights('param_vgg.hdf5')

model.fit(train_img_array, df_label_train, validation_data=(val_img_array, df_label_val), batch_size=32, epochs=6, verbose=1)



scores = model.evaluate(test_img_array, df_label_test, verbose=1)

print("test loss: ", scores[0])

print("test accuracy: ", scores[1])



model.summary()

from sklearn.metrics import confusion_matrix



prediction = model.predict(test_img_array)

prediction = np.argmax(prediction, axis=1)



tmp_array = df_label_test['NORMAL'].values

tmp_array2 = df_label_test['PNUEMONIA'].values

tmp_array = np.reshape(tmp_array, (tmp_array.shape[0], 1))

tmp_array2 = np.reshape(tmp_array2, (tmp_array2.shape[0],1))

tmp_array = np.concatenate([tmp_array, tmp_array2], axis=1)

tmp_array = np.argmax(tmp_array, axis=1)
from mlxtend.plotting import plot_confusion_matrix



cm = confusion_matrix(tmp_array, prediction)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)



plt.xticks(range(2), ['NORMAL', 'PNUEMONIA'], fontsize=16)

plt.yticks(range(2), ['NORMAL', 'PNUEMONIA'], fontsize=16)

plt.show()
recall = cm[0][0]/np.sum(cm[0])

print(recall)

precision = cm[0][0]/(cm[0][0]+cm[1][0])

print(precision)