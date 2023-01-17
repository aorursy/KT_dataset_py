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

import glob

import cv2

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt



from scipy import stats







%matplotlib inline

%config InlineBackend.figure_format = 'retina'
train_df = pd.read_csv('../input/landmark-retrieval-2020/train.csv')

train_df
sns.set()

plt.title('Training set: number of images per class(line plot)')

landmarks_fold = pd.DataFrame(train_df['landmark_id'].value_counts())

landmarks_fold.reset_index(inplace=True)

landmarks_fold.columns = ['landmark_id','count']

ax = landmarks_fold['count'].plot(logy=True, grid=True)

locs, labels = plt.xticks()

plt.setp(labels, rotation=30)

ax.set(xlabel="Landmarks", ylabel="Number of images")
sns.set()

landmarks_fold_sorted = pd.DataFrame(train_df['landmark_id'].value_counts())

landmarks_fold_sorted.reset_index(inplace=True)

landmarks_fold_sorted.columns = ['landmark_id','count']

landmarks_fold_sorted = landmarks_fold_sorted.sort_values('landmark_id')

ax = landmarks_fold_sorted.plot.scatter(\

     x='landmark_id',y='count',

     title='Training set: number of images per class(statter plot)')

locs, labels = plt.xticks()

plt.setp(labels, rotation=30)

ax.set(xlabel="Landmarks", ylabel="Number of images")
test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')

index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')
plt.rcParams["axes.grid"] = False

f, axarr = plt.subplots(4, 3, figsize=(24, 22))



curr_row = 0

for i in range(12):

    example = cv2.imread(test_list[i])

    example = example[:,:,::-1]

    

    col = i%4

    axarr[col, curr_row].imshow(example)

    if col == 3:

        curr_row += 1

            

#     plt.imshow(example)

#     plt.show()