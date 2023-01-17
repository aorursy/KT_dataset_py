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
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read the training data.





df = pd.read_csv('../input/global-wheat-detection/train.csv')

bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

for i, column in enumerate(['x', 'y', 'w', 'h']):

    df[column] = bboxs[:,i]

df.drop(columns=['bbox'], inplace=True)

df['x_center'] = df['x'] + df['w']/2

df['y_center'] = df['y'] + df['h']/2

df['classes'] = 0

from tqdm.auto import tqdm

import shutil as sh

df = df[['image_id','x', 'y', 'w', 'h','x_center','y_center','classes']]
df.head()
index = list(set(df.image_id))

len(index)
Image(filename='/kaggle/input/global-wheat-detection/train/b902a5132.jpg',width=600)
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

                row = mini[['classes','x_center','y_center','w','h']].astype(float).values

                row = row/1024

                row = row.astype(str)

                for j in range(len(row)):

                    text = ' '.join(row[j])

                    f.write(text)

                    f.write("\n")

            if not os.path.exists('convertor/fold{}/images/{}'.format(fold,path2save)):

                os.makedirs('convertor/fold{}/images/{}'.format(fold,path2save))

            sh.copy("../input/global-wheat-detection/{}/{}.jpg".format(source,name),'convertor/fold{}/images/{}/{}.jpg'.format(fold,path2save,name))
print(os.listdir("../input"))
# As i am running it for just trial(To save training time and GPU ) 

# So i am considering all the training factors to a limited extent.



# Play with all featuers and see their performance.





# !python train.py --img 1024 --batch 20 --epochs 10 --data ../input/yaml-file-for-data-model/wheat0.yaml --cfg ../input/yaml-file-for-data-model/yolov5x.yaml --name yolov5x_fold0_new







!python train.py --img 640 --batch 2 --epochs 1 --data ../input/yaml-file-for-data-model/wheat0.yaml --cfg ../input/yaml-file-for-data-model/yolov5x.yaml --name yolov5x_fold0_new



# ! pip install tree
# Start tensorboard

# Launch after you have started training

# logs save in the folder "runs"

# %load_ext tensorboard

# %tensorboard --logdir runs
# trained weights are saved by default in the weights folder

%ls weights/
!ls ./convertor
# create a prediction on validation data



!python ./detect.py --weights ./weights/last_yolov5x_fold0_new.pt --img 640 --conf 0.4 --source ./convertor/fold0/images/val2017
!ls /kaggle/working/inference/output/
# This will work from your end when you edit this notebook and run it.

# Image(filename='/kaggle/working/inference/output/42099cf54.jpg', width=400)
# This will work from your end when you edit this notebook and run it.

# Image(filename='/kaggle/working/inference/output/ad6e9eea2.jpg', width=400)