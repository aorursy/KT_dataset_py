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

from fastai.core import *
path = Path("/kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset")

data= ImageDataBunch.from_folder(path,train = '.',valid_pct = 0.2, ds_tfms = get_transforms(),size = 224 , num_workers= 4).normalize(imagenet_stats)
data.show_batch()
learn = cnn_learner(data,models.resnet101,metrics = accuracy)

learn.fit_one_cycle(10,1e-3,1e-2)