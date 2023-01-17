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
# train dataset load using pandas

tr_numeric = pd.read_csv('../input/Bosch_train_numeric.csv')

tr_numeric.describe()
tr_date = pd.read_csv('../input/Bosch_train_date.csv')

tr_date.describe()
tr_categorical = pd.read_csv('../input/Bosch_train_categorical.csv', dtype='category')

tr_categorical.describe()
# test dataset load using pandas

test_numeric = pd.read_csv('../input/Bosch_test_numeric.csv')

test_numeric.describe()
test_date = pd.read_csv('../input/Bosch_test_date.csv')

test_date.describe()
test_categorical = pd.read_csv('../input/Bosch_test_categorical.csv', dtype='category')

test_categorical.describe()