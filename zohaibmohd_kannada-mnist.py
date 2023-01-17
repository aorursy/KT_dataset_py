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
from fastai.tabular import * 

import pandas as pd



df = pd.read_csv("../input/Kannada-MNIST/train.csv")

path = '.'

procs = [FillMissing, Categorify, Normalize]

valid_idx = range(len(df)-2000, len(df))

dep_var = 'label'

cat_names = ['pixel0', 'pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5', 'pixel6', 'pixel7','pixel8','pixel9']

data = TabularDataBunch.from_df(path, df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)

learn = tabular_learner(data, layers=[200,100], metrics=accuracy)

learn.fit_one_cycle(10, 1e-2)
from fastai.widgets import ClassConfusion

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()