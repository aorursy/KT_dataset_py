# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from bokeh.plotting import figure, show, output_notebook
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 10.0) # set default size of plots
output_notebook()
train_dir = "../input"
train_file = "train.csv"

fbcheckin_train_tbl = pd.read_csv(os.path.join(train_dir, train_file))
fbcheckin_train_tbl = fbcheckin_train_tbl.sort_values(by="place_id")
sample_fbcheckin_train_tbl = fbcheckin_train_tbl[:10000]
plt.scatter(sample_fbcheckin_train_tbl["x"], sample_fbcheckin_train_tbl["y"], c=sample_fbcheckin_train_tbl["place_id"])
plt.title("Places sample distribution over (x,y)")


