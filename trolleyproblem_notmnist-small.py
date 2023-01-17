# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
os.listdir('../input/notmnist_small/notMNIST_small')
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
# %%time

# path = '../input/notmnist_large/notMNIST_large'

path = '../input/notmnist_small/notMNIST_small/'

# os.listdir(path)

# os.listdir('.')



fnames = []

for root, dir, files in os.walk(path):

    for f in files:

        if f.endswith('.png'):

            fnames.append(os.path.join(root, f))

# fnames

# fnames = get_image_files(path,recurse=True)

print(len(fnames))

fnames[:3]
def filter_out_bad_images(fnames):

    from matplotlib.pyplot import imread

    from tqdm import tqdm_notebook as tqdm

    from joblib import Parallel, delayed

    import multiprocessing  

    

    def mark_bad_image(f):

        try:

            im = imread(str(f))

        except Exception as e:

            print('Could not read:', f, ':', e, '- it\'s ok, skipping.')

            return f



#     PROPER FILTERING

    bad_images = Parallel(n_jobs=4)(delayed(mark_bad_image)(f) for f in tqdm(fnames))

    bad_images = [f for f in bad_images if f!=None]

    print(bad_images)

    

#     BAD ONES previously found

#     bad_images = ['../input/notmnist-tar/notmnist_large/notMNIST_large/A/RnJlaWdodERpc3BCb29rSXRhbGljLnR0Zg==.png', 

#                   '../input/notmnist-tar/notmnist_large/notMNIST_large/A/Um9tYW5hIEJvbGQucGZi.png', 

#                   '../input/notmnist-tar/notmnist_large/notMNIST_large/A/SG90IE11c3RhcmQgQlROIFBvc3Rlci50dGY=.png', 

#                   '../input/notmnist-tar/notmnist_large/notMNIST_large/D/VHJhbnNpdCBCb2xkLnR0Zg==.png', 

#                   '../input/notmnist-tar/notmnist_large/notMNIST_large/B/TmlraXNFRi1TZW1pQm9sZEl0YWxpYy5vdGY=.png']

    return [f for f in fnames if f not in bad_images]



fnames = filter_out_bad_images(fnames)

len(fnames)
# data = ImageDataBunch.from_folder(path,valid_pct=0.1)



np.random.seed(2)

pat = r'/([A-Z])/[^/]+png$'

data = ImageDataBunch.from_name_re(path, fnames, pat,valid_pct=0.2)

data.normalize()

data.classes
data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet50, metrics=error_rate,  model_dir="/tmp/model/")
learn.fit_one_cycle(4)
learn.save('stage-1')

learn.export('/kaggle/working/stg1.pkl')
learn.load('stage-1')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.load('stage-1')

learn.unfreeze()

learn.fit_one_cycle(6, max_lr=slice(1e-5,1e-3))
# learn.lr_find()

# learn.recorder.plot()
# learn.fit_one_cycle(6, max_lr=slice(1e-6,1e-4))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(25, figsize=(15,11))
interp.plot_confusion_matrix()
interp.most_confused()[:10]
model_path = learn.save('stage-2')

learn.export('/kaggle/working/stg2.pkl')