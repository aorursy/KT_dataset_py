# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import bq_helper

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data",dataset_name = "boneage-training-dataset")
#hacker_news.list_tables()

# Any results you write to the current directory are saved as output.
#dataset = pd.read_csv("../input/boneage-training-dataset.csv")
#dataset = pd.read_csv("../input/boneage-training-dataset.zip")

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
import numpy as np
from time import time
from time import sleep

folder = "../input/boneage-training-dataset/boneage-training-dataset"
folder_test = "../input/boneage-test-dataset/boneage-test-dataset"
#folder = "../input/regression_sample"

onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
onlyfiles_test = [f for f in os.listdir(folder_test) if os.path.isfile(os.path.join(folder_test, f))]

print("Working with {0} images".format(len(onlyfiles)))
print("Image examples: ")

for i in range(40, 42):
    print(onlyfiles[i])
    display(_Imgdis(filename=folder + "/" + onlyfiles[i], width=240, height=320))
    
    
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_files = []
test_files = []
i=0

for _file in onlyfiles:
    train_files.append(_file)
    
for _file in onlyfiles_test:
    test_files.append(_file)
    #print(train_files)
    #label_in_file = _file.find("_")
    #y_train.append(int(_file[0:label_in_file]))
print("Files in train_files: %d" % len(train_files))
print("Files in test_files: %d" % len(test_files))
#train_files[0]
#print(train_files[0:])
img_df = pd.DataFrame(data = train_files,  # This converts Array to Dataframe, with Index and Column names
                  index=None,
                  columns = None)
img_df_test = pd.DataFrame(data = test_files,  # This converts Array to Dataframe, with Index and Column names
                  index=None,
                  columns = None)

csv_df = pd.read_csv("../input/boneage-training-dataset.csv")
csv_df_test = pd.read_csv("../input/boneage-test-dataset.csv")

df_train = pd.concat([img_df,csv_df],axis = 1)  # Join two Dataframes, Finally
df_test = pd.concat([img_df_test,csv_df_test],axis = 1)

img_df = []
img_df_test = []
#pd.join(img_df,csv_df)
#img_df
#csv_df
df_train = df_train.rename(index=str, columns={0: "file"}) #Change name of Column from 0 to FIle
df_test = df_test.rename(index=str, columns={0: "file"})

df_y = df_train[['boneage']].copy()

df_train = df_train.drop(columns = ['boneage'],axis = 1)  # Dropped Y values from columns

#df_train
#df_test

image_width = 320
image_height = 240
ratio = 4

image_width = int(image_width / ratio)
image_height = int(image_height / ratio)

channels = 3
nb_classes = 1

dataset = np.ndarray(shape=(len(df_train), channels, image_height, image_width),dtype=np.float32)
dataset_test = np.ndarray(shape=(len(df_test), channels, image_height, image_width),dtype=np.float32)


i = 0
#print(folder + "/" + df_train['file'])
for _file in df_train['file']:
    #print(folder + "/" + _file)
    img = load_img(folder + "/" + _file,grayscale=False,target_size=[60,80],interpolation='nearest')  # this is a PIL image
    #img = load_img(folder + "/" + _file)  # this is a PIL image
    #print(img)
    img.thumbnail((image_width, image_height))
    # Convert to Numpy Array
    x = img_to_array(img)  
    x = x.reshape((3, 60, 80))
    #Normalize
    x = (x - 128.0) / 128.0
    dataset[i] = x
    i += 1
    if i % 250 == 0:
        print("%d images to array" % i)
print("All TRAIN images to array!")

j = 0
#print(folder + "/" + df_train['file'])
for _file in df_test['file']:
    #print(folder + "/" + _file)
    img = load_img(folder_test + "/" + _file,grayscale=False,target_size=[60,80],interpolation='nearest')  # this is a PIL image
    #img = load_img(folder + "/" + _file)  # this is a PIL image
    #print(img)
    img.thumbnail((image_width, image_height))
    # Convert to Numpy Array
    x = img_to_array(img)  
    x = x.reshape((3, 60, 80))
    #Normalize
    x = (x - 128.0) / 128.0
    dataset_test[j] = x
    j += 1
    if j % 250 == 0:
        print("%d images to array" % j)
print("All TEST images to array!")

df_train = []
df_test = []

#dataset_test.shape

# This will flatten the entire array of [3,120,160], and also reshape it with right dimention
# https://stackoverflow.com/questions/36967920/numpy-flatten-rgb-image-array
img_flat = dataset.flatten().reshape((12611,3*60*80))
img_flat_test = dataset_test.flatten().reshape((200,3*60*80))

img_flat.shape

# This converts Array to Dataframe, with Index and Column names
# https://stackoverflow.com/questions/20763012/creating-a-pandas-dataframe-from-a-numpy-array-how-do-i-specify-the-index-column
img_f = pd.DataFrame(data = img_flat[0:,1:],
                  index=None,
                  columns = None)
img_f_test = pd.DataFrame(data = img_flat_test[0:,1:],
                  index=None,
                  columns = None)
img_f_test.shape

# img_f = pd.DataFrame(data = img_flat[1:,1:],
#                   index=img_flat[1:,0],
#                   columns = img_flat[0,1:])
# img_f_test = pd.DataFrame(data = img_flat_test[1:,1:],
#                   index=img_flat_test[1:,0],
#                   columns = img_flat_test[0,1:])

#I think I got my dataFrame

img_flat = []
img_flat_test = []

#img_f_test

#img_f.dtypes.value_counts()
#One hot encoding OLD
# oneHot_train = pd.get_dummies(img_f)
# oneHot_test = pd.get_dummies(img_f_test)

#One hot encoding NEW
img_f = pd.get_dummies(img_f)
img_f_test = pd.get_dummies(img_f_test)

# img_f = []
# img_f_test = []

#final_train, final_test = oneHot_train.align(oneHot_test,join='left',axis=1)
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
# final_train = pd.DataFrame(my_imputer.fit_transform(oneHot_train))
# final_test = pd.DataFrame(my_imputer.transform(oneHot_test))
img_f = pd.DataFrame(my_imputer.fit_transform(img_f))
img_f_test = pd.DataFrame(my_imputer.transform(img_f_test))


# final_train.shape

from sklearn.model_selection import train_test_split
train_X,test_X, train_y,test_y = train_test_split(img_f.as_matrix(),df_y.as_matrix(), test_size=0.25)

print("Completed")
from xgboost import XGBRegressor

#my_model = XGBRegressor()
my_model = XGBRegressor(n_estimators = 1000,silence= False)
#my_model.fit(train_X,train_y, verbose = False)
my_model.fit(train_X,train_y,early_stopping_rounds=5,eval_set = [(test_X,test_y)], verbose = True)
predictions = my_model.predict(test_X)
from sklearn.metrics import mean_absolute_error
print("MAE : "+ str(mean_absolute_error(predictions,test_y)))
#test_y.describe()

