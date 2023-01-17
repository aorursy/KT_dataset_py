# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import tensorflow as tf

print(tf.__version__)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
data_train_file = "/kaggle/input/testdata/Test-Data.csv"     #Data.xlsx or Data.csv

data_test_file = "/kaggle/input/testdata/Test-Data.csv"



df_train = pd.read_csv(data_train_file)

df_train = pd.read_csv(data_test_file)
df_train.head()