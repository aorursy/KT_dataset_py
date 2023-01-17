import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np 





# import useful tools

from glob import glob

from PIL import Image

import cv2



# import data visualization

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import seaborn as sns



from bokeh.plotting import figure

from bokeh.io import output_notebook, show, output_file

from bokeh.models import ColumnDataSource, HoverTool, Panel

from bokeh.models.widgets import Tabs



from tqdm.auto import tqdm

import shutil as sh



# import data augmentation

import albumentations as albu
!ls ../input/count-the-number-of-faces-present-in-an-image/train
# Setup the paths to train and test images

train=pd.read_csv('../input/count-the-number-of-faces-present-in-an-image/train/train.csv')

test=pd.read_csv('../input/count-the-number-of-faces-present-in-an-image/test.csv')



Images='../input/count-the-number-of-faces-present-in-an-image/train/image_data/'

# Glob the directories and get the lists of train and test images

img = glob(Images + '*')

# Compute at the number of images:

print('Total Number of images is {}'.format(len(img)))
print('Number of image in train data are {}'.format(train.shape[0]))

train.head()
print('Number of image in test data are {}'.format(test.shape[0]))

test.head()
bbox=pd.read_csv('../input/count-the-number-of-faces-present-in-an-image/train/bbox_train.csv')

bbox.head()
# Merge all train images with the bounding boxes dataframe



train_images = train.merge(bbox, on='Name', how='left')
print(train_images.isnull().sum())

print(train_images.shape)

train_images
### Let's plot some image examples:



train_images.iloc[2].Name
# First we store all the box dimensions.

def get_all_bboxes(df, image_id):

    image_bboxes = df[df.Name == image_id]

    

    bboxes = []

    for _,row in image_bboxes.iterrows():

        bboxes.append((row.xmin, row.ymin, row.xmax, row.ymax))

        

    return bboxes



# function for box representation on the image.



def plot_image_with_box(df, rows=3, cols=4, title='Face count images'):

    fig, axs = plt.subplots(rows, cols, figsize=(20,15))

    for row in range(rows):

        for col in range(cols):

            idx = np.random.randint(len(df), size=1)[0]

            img_id = df.iloc[idx].Name

            

            img = Image.open(Images + img_id)

            axs[row, col].imshow(img)

            

            bboxes = get_all_bboxes(df, img_id)

            

            for bbox in bboxes:

                rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='g',facecolor='none')

                axs[row, col].add_patch(rect)

            

            axs[row, col].axis('off')

            

    plt.suptitle(title)
plot_image_with_box(train_images)
train
# compute the number of bounding boxes per train image

# train_images['count'] = train_images.loc[:,train_images.columns !='HeadCount'].apply(lambda row: 1 if np.isfinite(row.width) else 0, axis=1)





# train_images_count = train_images.loc[:,train_images.columns !='HeadCount'].groupby('Name').sum().reset_index()
# train_images_count['HeadCount']=train['HeadCount']

# train_images_count.head()
# len(train_images_count.Name.unique())
# See this article on how to plot bar charts with Bokeh:

# https://towardsdatascience.com/interactive-histograms-with-bokeh-202b522265f3



def hist_hover(dataframe, column, colors=["#94c8d8", "#ea5e51"], bins=30, title=''):

    hist, edges = np.histogram(dataframe[column], bins = bins)

    

    hist_df = pd.DataFrame({column: hist,

                             "left": edges[:-1],

                             "right": edges[1:]})

    hist_df["interval"] = ["%d to %d" % (left, right) for left, 

                           right in zip(hist_df["left"], hist_df["right"])]



    src = ColumnDataSource(hist_df)

    plot = figure(plot_height = 400, plot_width = 600,

          title = title,

          x_axis_label = 'Faces in image',

          y_axis_label = "Count")    

    plot.quad(bottom = 0, top = column,left = "left", 

        right = "right", source = src, fill_color = colors[0], 

        line_color = "#35838d", fill_alpha = 0.7,

        hover_fill_alpha = 0.7, hover_fill_color = colors[1])

        

    hover = HoverTool(tooltips = [('Interval', '@interval'),

                              ('Count', str("@" + column))])

    plot.add_tools(hover)

    

    output_notebook()

    show(plot)
hist_hover(train_images, 'HeadCount', title='Number of faces per image')
train_images.head()
df=train_images

df.head()
df['x_center'] = df['xmin'] + df['width']/2

df['y_center'] = df['ymin'] + df['height']/2

df['classes'] = 0





df['image_id']=df['Name'].str.replace('.jpg','')



df = df[['image_id','xmin', 'ymin', 'width', 'height','x_center','y_center','classes']]
df.head()
from IPython.display import Image, clear_output  # to display images
# import required dependencies



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from tqdm.auto import tqdm

import shutil as sh



import matplotlib.pyplot as plt



%matplotlib inline
!git clone https://github.com/AIVenture0/yolov5.git
# check for the cloned repo

!ls -R
# move all the files of YOLOv5 to current working directory

!mv yolov5/* ./
# check for all the files in the current working directory

!ls
!pip install -r requirements.txt
# # read the training data.





# df = pd.read_csv('../input/global-wheat-detection/train.csv')

# bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

# for i, column in enumerate(['x', 'y', 'w', 'h']):

#     df[column] = bboxs[:,i]

# df.drop(columns=['bbox'], inplace=True)

# df['x_center'] = df['x'] + df['w']/2

# df['y_center'] = df['y'] + df['h']/2

# df['classes'] = 0

# from tqdm.auto import tqdm

# import shutil as sh

# df = df[['image_id','x', 'y', 'w', 'h','x_center','y_center','classes']]
# count

index = list(set(df.image_id))

len(index)

# code to transform the dataset.



source = 'train'

if True:

    for fold in [0]:

        val_index = index[len(index)*fold//5:len(index)*(fold+1)//5]

        for name,mini in tqdm(df.groupby('image_id')):

            if name in val_index:

                path2save = 'val2017/'

            else:

                path2save = 'train2017/'

            if not os.path.exists('convertor/fold{}/labels/'.format(fold)+path2save):

                os.makedirs('convertor/fold{}/labels/'.format(fold)+path2save)

            with open('convertor/fold{}/labels/'.format(fold)+path2save+name+".txt", 'w+') as f:

                row = mini[['classes','x_center','y_center','width','height']].astype(float).values

                row = row/1024

                row = row.astype(str)

                for j in range(len(row)):

                    text = ' '.join(row[j])

                    f.write(text)

                    f.write("\n")

            if not os.path.exists('convertor/fold{}/images/{}'.format(fold,path2save)):

                os.makedirs('convertor/fold{}/images/{}'.format(fold,path2save))

            sh.copy("../input/count-the-number-of-faces-present-in-an-image/{}/image_data/{}.jpg".format(source,name),'convertor/fold{}/images/{}/{}.jpg'.format(fold,path2save,name))
print(os.listdir("../input/count-the-number-of-faces-present-in-an-image/train"))
# !ls ./convertor



!ls ./convertor/fold0/labels/train2017/12433.txt
# As i am running it for just trial(To save training time and GPU ) 

# So i am considering all the training factors to a limited extent.



# Play with all featuers and see their performance.





# !python train.py --img 1024 --batch 20 --epochs 10 --data ../input/yaml-file-for-face-count-data-model/face_count.yaml --cfg ../input/yaml-file-for-face-count-data-model/yolov5x.yaml --name yolov5x_fold0_new





!python ./train.py --img 640 --batch 3 --epochs 20 --data ../input/yaml-file-for-face-count-data-model/face_count.yaml --cfg ../input/yaml-file-for-face-count-data-model/yolov5x.yaml --name yolov5x_fold0_new
# trained weights are saved by default in the weights folder

%ls weights/


!python ./detect.py --weights ./weights/last_yolov5x_fold0_new.pt --img 640 --conf 0.4 --source ./convertor/fold0/images/val2017
# This will work from your end when you edit this notebook and run it.

Image(filename='/kaggle/working/inference/output/16800.jpg', width=400)
Image(filename='/kaggle/working/inference/output/10185.jpg', width=400)
Image(filename='/kaggle/working/inference/output/10118.jpg', width=400)