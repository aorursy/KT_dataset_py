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


import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



data = pd.read_csv("../input/autompg-dataset/auto-mpg.csv")
data
data.describe()
data.isnull().any()
data.horsepower.unique()

data = data.replace("?", np.NaN)

data.horsepower.unique()

data.isnull().any()
length=[85,15,24,58,26,59,69,96]

breadth=[48,25,36,51,59,68,36,47]

df=pd.DataFrame({"length":length,"breadth":breadth})

df

df["area"]=df["length"]*df["breadth"]

df





area