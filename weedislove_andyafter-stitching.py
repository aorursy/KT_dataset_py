# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
from PIL import ImageFilter
import multiprocessing
import random; random.seed(2016);
import cv2
import re
import os, glob

sample_sub = pd.read_csv('../input/sample_submission.csv')
train_files = pd.DataFrame([[f,f.split("/")[3].split(".")[0].split("_")[0],f.split("/")[3].split(".")[0].split("_")[1]] for f in glob.glob("../input/train_sm/*.jpeg")])
train_files.columns = ['path', 'group', 'pic_no']
test_files = pd.DataFrame([[f,f.split("/")[3].split(".")[0].split("_")[0],f.split("/")[3].split(".")[0].split("_")[1]] for f in glob.glob("../input/test_sm/*.jpeg")])
test_files.columns = ['path', 'group', 'pic_no']
print(len(train_files),len(test_files),len(sample_sub))
train_images = train_files[train_files["group"]=='set107']
train_images = train_images.sort_values(by=["pic_no"], ascending=[1]).reset_index(drop=True)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
plt.subplots_adjust(wspace=0, hspace=0)
i_ = 0
a = []
for l in train_images.path:
    im = cv2.imread(l)
    plt.subplot(5, 2, i_+1).set_title(l)
    plt.hist(im.ravel(),256,[0,256]); plt.axis('off')
    a.append([im.mean(),im.max(),im.min()])
    plt.subplot(5, 2, i_+2).set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 2
print(a)
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.