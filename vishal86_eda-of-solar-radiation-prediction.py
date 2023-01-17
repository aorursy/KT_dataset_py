# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

##import (library) as (give the library a nickname/alias)

import matplotlib.pyplot as plt

import pandas as pd #this is how I usually import pandas



from numpy.random import randn

from matplotlib import rcParams

import seaborn as sb



# Enable inline plotting

%matplotlib inline
df=pd.read_csv('../input/SolarPrediction.csv')
df.columns
df['Radiation'].hist()