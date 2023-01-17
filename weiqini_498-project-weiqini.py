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
from os.path import join, isfile
from os import path, scandir, listdir

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import geopandas as gp
from shapely.geometry import Point

import gc
df_train = pd.read_csv('/kaggle/input/pollutionvision/train_data.csv')

df_train.head()

#df_train.procedure_id.groupby(df.patient_id).nunique().hist();

#print(df_train.shape)

#(64961, 20)


df_train_clean=df_train[['Temp(C)','Pressure(kPa)','Image_file','Wind_Speed','Distance_to_Road','Camera_Angle','Elevation','Total']].loc[(df_train['Errors'] == 0) & (df_train['Dead Time'] < 0.01)]
df_train_clean

#.head()

#print(df_train.shape)
#(60868, 8)
df_train_sort=df_train_clean.sort_values(by=['Image_file'])

df_train_sort['image date'] = df_train_sort['Image_file'].str[5:9]

df_train_sort

#df_train_sortbydate=df_train_sort.groupby(by='image date')

#df_train_sortbydate
df_test = pd.read_csv('/kaggle/input/pollutionvision/test_data.csv')

df_test.head()

#print(df_test.shape)
#(7200, 20)
df_test_clean=df_test[['Temp(C)','Pressure(kPa)','Image_file','Wind_Speed','Distance_to_Road','Camera_Angle','Elevation']].loc[(df_test['Errors'] == 0) & (df_test['Dead Time'] < 0.01)]
df_test_clean

#.head()

#print(df_test_clean.shape)
#(7193, 7)

#df_test_clean

 
df_sample = pd.read_csv('/kaggle/input/pollutionvision/sample.csv')
print(df_sample.shape)
#for training data
df_train_clean.describe()
#for test data
df_test_clean.describe()
df_train_corr=df_train_clean.drop(columns='Image_file')
corrMatrix = df_train_corr.corr()
print (corrMatrix)

plt.matshow(corrMatrix)
plt.xticks(range(df_train_corr.shape[1]), df_train_corr.columns, fontsize=14, rotation=45)
plt.yticks(range(df_train_corr.shape[1]), df_train_corr.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
#plt.title('Correlation Matrix', fontsize=16);

plt.show()
df_train_image=df_train_sort

#extract image data and the order in each date in "image date" and "image number"
df_train_image['image number'] = df_train_image['Image_file'].str.split('_').str[1]
df_train_image['image number'] = df_train_image['image number'].str.split('.').str[0].astype(int)

#sort df by the image data and the order
df_train_image=df_train_image.sort_values(by=['image date','image number'])

df_train_image


df_train_0608=df_train_image.loc[df_train_image['image date'] == '0608']

df_train_0608
                        
df_train_0608_describe=df_train_0608.describe()

df_train_0608_describe

#get the dates list

#Image_date_list = df_train_image["image date"].unique().tolist()
#Image_date_list

# 1 make dict 

#dict_of_dates = dict(iter(df_train_image.groupby('image date')))
#print(dict_of_dates)
#dict_of_dates

# 2 make for loop     
    
for i,g in df_train_image.groupby('image date'):
    globals()['df_train_' + str(i)] =  g
    
df_train_0611
#df_train_0611.describe()

#now we have generate the sebset of df by dates (0608-0810)
#the df_train_mean has mean and std info for "T,P,Wind_speed,...,Total" of different dates
df_train_mean = df_train_image.groupby('image date').agg([np.mean, np.std])
df_train_mean



Total_mean = df_train_mean['Total'] 
#Total_mean

Total_mean.plot(kind = "barh", y = "mean", legend = False, 
            title = "Mean Total", xerr = "std")
#get the pathway of files in list

def list_all_files(location='../input/pollutionvision/frames/frames/', pattern=None, recursive=True):
    """
    This function returns a list of files at a given location (including subfolders)
    
    - location: path to the directory to be searched
    - pattern: part of the file name to be searched (ex. pattern='.csv' would return all the csv files)
    - recursive: boolean, if True the function calls itself for every subdirectory it finds
    """
    subdirectories= [f.path for f in scandir(location) if f.is_dir()]
    files = [join(location, f) for f in listdir(location) if isfile(join(location, f))]
    if recursive:
        for directory in subdirectories:
            files.extend(list_all_files(directory))
    if pattern:
        files = [f for f in files if pattern in f]
    return files

x=list_all_files()

print(len(x))
from PIL import Image

im1 = Image.open("/kaggle/input/pollutionvision/frames/frames/video06112020_10.jpg")
im2 = Image.open("/kaggle/input/pollutionvision/frames/frames/video08102020_3159.jpg")

#print(im.format, im.size, im.mode)
#JPEG (1280, 720) RGB

#06112020_6
#08102020_3159

plt.imshow(im1)