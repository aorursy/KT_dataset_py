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
TRAIN_PATH = os.path.join("../input/", "BBC News Train.csv")



#load the datat using pandas

df = pd.read_csv(TRAIN_PATH)

df.head()
# changes categories into numbers

df['category_id'] = df['Category'].factorize()[0]



#show the first 10 entires of category_id

df['category_id'][0:10]
# Create a new pandas dataframe "category_id_df", which only has unique Categories, also sorting this list in order of category_id values

category_id_df = df[['Category', 'category_id']].drop_duplicates().sort_values('category_id')

category_id_df #prints output
# Create a dictionary ( python datastructure - like a lookup table/Map) that 

# can easily convert category names into category_ids and vice-versa

category_to_id = dict(category_id_df.values) #links Category (key) to category_id (value)

id_to_category = dict(category_id_df[['category_id', 'Category']].values) #links category_id (key) to Category (value)