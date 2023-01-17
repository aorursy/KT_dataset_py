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
COLLISIONS_CSV = '../input/nypd-motor-vehicle-collisions.csv'
METADATA_JSON = '../input/socrata_metadata.json'
DATADICT = '../input/Collision_DataDictionary.xlsx'
collisions = pd.read_csv(COLLISIONS_CSV)
collisions.head()
collisions.tail()
collisions.describe()
collisions.info()
print(f'Number of Columns: {len(collisions.columns)}')
latest_date = collisions.iloc[0].DATE
earliest_date = collisions.iloc[-1].DATE
latest_date
earliest_date
injured_df = collisions[collisions['NUMBER OF PERSONS INJURED'] > 0]
killed_df = collisions[collisions['NUMBER OF PERSONS KILLED'] > 0]
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, figsize=(12,8))
injured_df.BOROUGH.value_counts().plot(kind='bar', color='blue', position=0, ax=ax)
killed_df.BOROUGH.value_counts().plot(kind='bar', color='red', position=1, ax=ax)
ax.set_title('Number of persons injured and killed per borough', fontsize=20, fontweight='bold')
ax.set_ylabel('Number of persons injured/killed')
plt.show()
fig = plt.figure(figsize=(12,8))
bx = fig.add_subplot(111)
bx2 = bx.twinx()
injured_df.BOROUGH.value_counts().plot(kind='bar', color='blue', width = 0.4, position=0, ax=bx)
killed_df.BOROUGH.value_counts().plot(kind='bar', color='red', width = 0.4, position=1, ax=bx2)
ax.set_title('Number of persons injured and killed per borough', fontsize=20, fontweight='bold')
ax.set_ylabel('Number of persons injured/killed')
plt.show()
