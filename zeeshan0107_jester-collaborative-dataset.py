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
dfJokes=pd.read_csv('../input/JokeText.csv')
dfJokes.head()
dfRatings1=pd.read_csv('../input/UserRatings1.csv')
dfRatings1.head()
dfRatings2=pd.read_csv('../input/UserRatings2.csv')
dfRatings2.head()
#MergedRatings = pd.merge(dfRatings,dfRatings2)
#MergedRatings.head()
#len(MergedRatings.columns)
#MergedRatings = MergedRatings.fillna(0)

dfRatings1 = dfRatings1.fillna(0)
dfRatings1 