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
data=pd.read_csv('../input/chess/games.csv')
data.shape
data.describe()
data.info()
import matplotlib.pyplot as plt
data.isnull().any().sum()
data[['created_at','black_rating','turns','last_move_at','opening_ply']].hist()
plt.boxplot(data['created_at'])
plt.boxplot(data['black_rating'])
plt.boxplot(data['turns'])
plt.boxplot(data['last_move_at'])
plt.boxplot(data['white_rating'])