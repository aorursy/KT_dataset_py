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
data = pd.read_csv('/kaggle/input/ncaa-women-538-team-ratings/538ratingsWomen.csv')
data.head()
data.shape
print('The number of unique teams is ', data.TeamName.nunique())
pd.DataFrame(data.TeamName.value_counts().head(20)).reset_index(drop = False).rename(columns = {'index' : 'TeamName', 'TeamName': 'No of entries'})
rating = (data.groupby('TeamName')['538rating'].sum().reset_index().sort_values(by = '538rating', ascending = False).head(20)).reset_index(drop = True)

rating