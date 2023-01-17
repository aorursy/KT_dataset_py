# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import pandas as pd

# Any results you write to the current directory are saved as output.
# Load Train and test dataset

# Overview of train dataset
df_train = pd.read_csv("../input/train.csv")
print("Train dataset loaded", format(str(df_train.shape[0])))
# overview of train data and columns
df_train.head()
# Load and explore test dataset
df_test = pd.read_csv("../input/test.csv")
print("test data loaded size: ", format(str(df_test.shape[0])))

# Display a sample
df_test.head()