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
data=pd.read_csv('../input/pokemon.csv')
data.sample(10)
data.info()
data.type1.value_counts().plot.bar()
data.hp.value_counts().sort_index().plot.line()
data.weight_kg.plot.hist()
data.plot.scatter(x='attack',y='defense')
data.plot.hexbin(x='attack',y='defense',gridsize=15)
data.groupby(['is_legendary', 'generation']).mean()[['attack','defense']].plot.bar(stacked=True)
data.groupby('generation').mean()[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].plot.line()