# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from fastai import *

from fastai.vision import *

#classes = ['BenZ','BMW','Audi']
#folder = 'BenZ'

#file = 'benz'
#path = Path('data/')

#dest = path/folder

#dest.mkdir(parents=True, exist_ok=True)
#!cp ../input/* {path}/
#!pwd

#download_images(path/file, dest, max_pics=200)
#folder = 'BMW'

#file = 'bmw'
#path = Path('data/bba')

#dest = path/folder

#dest.mkdir(parents=True, exist_ok=True)
#download_images(path/file, dest, max_pics=200)
#folder = 'Audi'

#file = 'audi'
#path = Path('data/bba')

#dest = path/folder

#dest.mkdir(parents=True, exist_ok=True)
#download_images(path/file, dest, max_pics=200)
# for c in classes:

#     print(c)

#     verify_images(path/c, delete=True, max_size=500)
!mkdir data
!cp -r /kaggle/input/bba-tar/bba /kaggle/working/data
path = Path('data/bba')
src = ImageList.from_folder(path).random_split_by_pct(0.2, seed=2)
 #np.random.seed(42)

 #data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.5,

#         ds_tfms=get_transforms(), size=64, num_workers=0).normalize(imagenet_stats)
tfms = get_transforms()
def get_data(size, bs, padding_mode='reflection'):

    return (src.label_from_folder()

           .transform(tfms, size=size, padding_mode=padding_mode)

           .databunch(bs=bs).normalize(imagenet_stats))
data = get_data((180,250),8)
data.show_batch(rows=3, figsize=(10,10))
learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(40)
learn.save('stage-1')
learn.load('stage-1');
learn.unfreeze()
#learn.lr_find()
#learn.recorder.plot()
learn.fit_one_cycle(40)
#learn.lr_find()

#learn.recorder.plot()
learn.save('stage-2')

learn.load('stage-2');
data = get_data((360,500),8)
learn = create_cnn(data, models.resnet50, metrics=error_rate).load('stage-2');
#learn.lr_find()
learn.fit_one_cycle(40)
learn.save('big-stage1')
learn.unfreeze()
learn.fit_one_cycle(45)

learn.save('big-stage2')
#interp = ClassificationInterpretation.from_learner(learn)


#interp.plot_confusion_matrix()
#from fastai.widgets import *
#ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=100)
#ImageCleaner(ds, idxs, path)