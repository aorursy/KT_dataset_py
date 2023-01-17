import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport

from pandas_summary import DataFrameSummary

import cv2

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import random

import math

from tqdm import tqdm

import seaborn as sns

from os import listdir

from os.path import isfile, join

%matplotlib inline



path = '../input/global-wheat-detection/'

TRAIN_IMAGES_PATH = path+'train/'
train = pd.read_csv(path+'train.csv')

sub = pd.read_csv(path+'sample_submission.csv')
def plot_images(dataframe, show_bb = True, image_count = 16,idxs = [], image_label='image_id', bbox_label='bbox', n = 3):

    """This function can display any no of image in the grid size of 4 by no_of_images/4.

        Usage:-

            dataframe = The dataframe containing images and bounding boxes grouped into single list.

            show_bb = Show bounding boxes or not.

            imge_count = No of images to display (multiple of n).

            idxs =  If u want to pass your own indexes for ploting use this else pass [] an empty array. It will automatically select random images.

            image_label = Name of the column containing images in dataframe.

            bbox_label = Name of the column containing list of all bounding boxes per image in dataframe.

            n = Number of images per row.



        [NOTE]: If you want to convert stock train dataframe to desired format use the clean_data method. 

    """

    size = len(dataframe)

    image_count = image_count + (image_count % n)

    if len(idxs)==0:

        create_idx = True

    else:

        create_idx = False

        

    row_count = (int) (image_count / n)

    fig, ax = plt.subplots(row_count, n, figsize=(20,10))

    for i in range(image_count):

        x = (int)(i/n)

        y = i%n

        if create_idx:

            idx = random.randint(0, size-1)

            idxs.append(idx)

        else:

            idx = idxs[i]

        input_row = dataframe.iloc[idx]

        tuple_index = (x,y) if row_count > 1 else y

        ax[tuple_index].imshow(cv2.imread(TRAIN_IMAGES_PATH + input_row[image_label]))

        ax[tuple_index].set_title(input_row['image_id'])

        if show_bb:

            try:

                bbs = input_row[bbox_label]

                for bbox in bbs:    

                    rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')

                    ax[tuple_index].add_patch(rect)

            except:

                pass

    fig.show()

    return idxs





def enlarge_image(dataframe, idx = -1, show_bb = True,image_label='image_id', bbox_label='bbox'):

    """This function is used to enlarge single image with or without bounding boxes.

        Usage:-

            dataframe = The dataframe containing images and bounding boxes grouped into single list.

            idx = The index of the image to be displayed. -1(default) means random selection.

            show_bb = Show bounding boxes or not.

            image_label = Name of the column containing images in dataframe.

            bbox_label = Name of the column containing list of all bounding boxes per image in dataframe.

        

        [NOTE]: If you want to convert stock train dataframe to desired format use the clean_data method. 

    """

    fig, ax = plt.subplots(figsize=(15,15))

    size = len(dataframe[dataframe['source']!='not_specified'])

    if idx==-1:

        idx = random.randint(0, size-1)

    input_row = dataframe.iloc[idx]

    tuple_index = (0,0)

    ax.imshow(cv2.imread(TRAIN_IMAGES_PATH + input_row[image_label]))

    ax.set_title(input_row[image_label])

    if show_bb:

        try:

            bbs = input_row[bbox_label]

            for bbox in bbs:    

                rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')

                ax.add_patch(rect)

        except:

            pass

    fig.show()

    



def clean_data(train):

    dic = {}

    imgs = []

    bbs = []

    srcs = []

    tmp = []

    tn='---'

    for i in tqdm(train.iterrows()):

        img,_,_,bb,s = i[1]

        if tn=='---':

            tn = img

        elif tn == img:

            tmp.append(list(map(math.floor, list(map(float,bb.replace('[',"").replace(']',"").split(','))))))

        else:

            imgs.append(tn+'.jpg')

            bbs.append(tmp)

            srcs.append(s)

            tn = img

            tmp=[]

            tmp.append(list(map(math.floor, list(map(float,bb.replace('[',"").replace(']',"").split(','))))))

    imgs.append(tn+'.jpg')

    bbs.append(tmp)

    srcs.append(s)

    dic['image_id']=imgs

    dic['bbox'] = bbs

    dic['source'] = srcs

    train_clean = pd.DataFrame(dic)

    return train_clean
train.head()
dfs = DataFrameSummary(train)



dfs.columns_stats
train_clean = clean_data(train)





"""Adding unlabeled images from train directory."""

dic = {}

onlyfiles = [f for f in listdir(path+'train') if isfile(join(path+'train', f))]

unlabeled = list(set(onlyfiles) - set(train_clean['image_id']))

dic['image_id'] = unlabeled

dic['bbox'] = [[] for i in range(len(unlabeled))]

dic['source'] = ['not_specified' for i in range(len(unlabeled))]

temp_clean = pd.DataFrame(dic)

train_clean = pd.concat([train_clean,temp_clean])
train_clean.head()
counts = dict(train_clean['source'].value_counts())



fig, ax = plt.subplots(figsize=(8,8));

wedges, texts, autotexts = ax.pie(list(counts.values()), autopct='%1.1f%%',

        shadow=True, startangle=90);

ax.legend(wedges, list(counts.keys()),

          title="Sources",

          loc="center",

          bbox_to_anchor=(1, 0, 0.5, 1));



plt.setp(autotexts, size=15);



ax.set_title("Data Distribution based on Sources");

plt.show();

sns.countplot(x='source', data=train_clean);
profile = ProfileReport(train, title='Report',progress_bar = False);

profile.to_widgets()
enlarge_image(train_clean)
idxs = plot_images(train_clean, show_bb = False, image_count = 3);

plot_images(train_clean, show_bb = True,idxs = idxs, image_count = 3);
plot_images(train_clean[train_clean['source']=='not_specified'], idxs = [],show_bb = True, image_count = 6);
idxs = plot_images(train_clean[train_clean['source']=='arvalis_1'], show_bb = False,idxs = [], image_count = 3);

plot_images(train_clean[train_clean['source']=='arvalis_1'], idxs = idxs, show_bb = True, image_count = 3);
idxs = plot_images(train_clean[train_clean['source']=='arvalis_2'], show_bb = False,idxs = [], image_count = 3);

plot_images(train_clean[train_clean['source']=='arvalis_2'], idxs = idxs, show_bb = True, image_count = 3);
idxs = plot_images(train_clean[train_clean['source']=='arvalis_3'], show_bb = False,idxs = [], image_count = 3);

plot_images(train_clean[train_clean['source']=='arvalis_3'], idxs = idxs, show_bb = True, image_count = 3);
idxs = plot_images(train_clean[train_clean['source']=='ethz_1'], show_bb = False,idxs = [], image_count = 3);

plot_images(train_clean[train_clean['source']=='ethz_1'], idxs = idxs, show_bb = True, image_count = 3);
idxs = plot_images(train_clean[train_clean['source']=='rres_1'], show_bb = False,idxs = [], image_count = 3);

plot_images(train_clean[train_clean['source']=='rres_1'], idxs = idxs, show_bb = True, image_count = 3);
idxs = plot_images(train_clean[train_clean['source']=='usask_1'], show_bb = False,idxs = [], image_count = 3);

plot_images(train_clean[train_clean['source']=='usask_1'], idxs = idxs, show_bb = True, image_count = 3);
idxs = plot_images(train_clean[train_clean['source']=='inrae_1'], show_bb = False,idxs = [], image_count = 3);

plot_images(train_clean[train_clean['source']=='inrae_1'], idxs = idxs, show_bb = True, image_count = 3);