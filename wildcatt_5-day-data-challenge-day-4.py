# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/20170308hundehalter.csv")
data.head()
breeds = data["RASSE1"]
# import seaborn and alias it as sns
import seaborn as sns

# make a barplot from a column in our dataframe
sns.countplot(data["RASSE1"]).set_title("Common Dog Breeds in Zurich")