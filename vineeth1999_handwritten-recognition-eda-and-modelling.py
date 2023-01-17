from IPython.display import HTML
HTML('<center><iframe width="560" height="315" src="https://www.youtube.com/embed/jO-1rztr4O0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')
import os
from os import listdir
import pandas as pd
import numpy as np
import glob
import tqdm
from typing import Dict
import matplotlib.pyplot as plt
%matplotlib inline

#plotly
!pip install chart_studio
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

#color
from colorama import Fore, Back, Style

import seaborn as sns
sns.set(style="whitegrid")

#cv2
import cv2

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# Settings for pretty nice plots
plt.style.use('fivethirtyeight')
plt.show()
list(os.listdir("../input/handwriting-recognition"))
IMAGE_PATH = "../input/handwriting-recognition"

train_df = pd.read_csv('../input/handwriting-recognition/written_name_train_v2.csv')
val_df = pd.read_csv('../input/handwriting-recognition/written_name_validation_v2.csv')
test_df = pd.read_csv('../input/handwriting-recognition/written_name_test_v2.csv')

print(Fore.YELLOW + 'Training data shape: ',Style.RESET_ALL,train_df.shape)
print(Fore.YELLOW + 'Validation data shape: ',Style.RESET_ALL,val_df.shape)
train_df.head(5)
train_df.groupby(['IDENTITY']).count()['FILENAME'].to_frame()
# Null values and Data types
print(Fore.YELLOW + 'Train Set !!',Style.RESET_ALL)
print(train_df.info())
print('-------------')
print(Fore.YELLOW + 'Validation Set !!',Style.RESET_ALL)
print(val_df.info())
print('-------------')
print(Fore.BLUE + 'Test Set !!',Style.RESET_ALL)
print(test_df.info())
train_df.isna().sum()
val_df.isna().sum()
test_df.isna().sum()
columns = train_df.keys()
columns = list(columns)
print(columns)
train_df['IDENTITY'].value_counts()
train_df['IDENTITY'].iplot(kind='hist',xTitle='NAME',linecolor='black',opacity=0.8,color='#FB8072',bargap=0.5,gridcolor='white',title='Distibution of names in training set')
def show_image(image,file):
    file_size = os.path.getsize(file)
    print(Fore.BLUE + "Image size.......:",Style.RESET_ALL," {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=image.shape[0], cols=image.shape[1], size=file_size))
    plt.figure(figsize=(20, 40))
    plt.imshow(image, cmap='gray')
    plt.show()
for file_path in glob.glob('../input/handwriting-recognition/train_v2/train/*.jpg'):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    show_image(image,file_path)
    break # Comment this out to see all
image = cv2.imread('../input/handwriting-recognition/train_v2/train/TRAIN_00001.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20, 40))
plt.imshow(image, cmap='jet')
import pandas_profiling as pdp
profile_train_df = pdp.ProfileReport(train_df)
profile_train_df
profile_test_df = pdp.ProfileReport(test_df)
profile_val_df = pdp.ProfileReport(val_df)
profile_test_df
profile_val_df