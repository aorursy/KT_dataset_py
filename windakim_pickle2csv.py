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
import csv

from six.moves import cPickle as pickle

import numpy as np
path = "../input/Thesis/ag_news.p"
def main(path_pickle,path_csv):



    x = []

    with open(path_pickle,'r') as f:

        x = pickle.load(f)



    with open(path_csv,'w') as f:

        writer = csv.writer(f)

        for line in x: writer.writerow(line)