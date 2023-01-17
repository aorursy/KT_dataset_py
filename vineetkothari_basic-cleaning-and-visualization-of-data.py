# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from mpl_toolkits.basemap import Basemap

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
state_df=pd.read_csv('../input/StatewiseTreeCover.csv')
state_df.head()
state_df.describe() #describe the numerical value
state_df['Tree Cover - Per cent of GA'].value_counts()#for non numerical value
#plotting histogram

state_df['Tree Cover - Per cent of GA'].hist(bins=50)

    
state_df.head().boxplot(column='Tree Cover - Per cent of GA',by='State/ Uts')#plotting to state maximum tree cover
from mpl_toolkits.basemap import Basemap

m = Basemap(projection='tmerc')