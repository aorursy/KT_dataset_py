

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
# Read the dataset into a pandas dataframe
mdf = pd.read_csv('../input/WorldCups.csv')
mdf
# SO as you know data science can answer your questions related to availble dataset
# Let's see top 10 world cup goals scored (year wise)
plt.figure(figsize=(15,8))
sns.barplot(x = 'Year' , y = 'GoalsScored', data = mdf)

# Let's see winner teams over the years 
mdf.Winner.value_counts().plot(kind='bar',figsize=(8,8))
#and which team was runner up most of teh time 
# Here i am changing the Column name becuase it was giving errors
mdf.rename(columns={'Runners-Up': 'R'}, inplace=True)
mdf.info()
# Runner up team
mdf.R.value_counts().plot(kind='bar',figsize=(8,8))