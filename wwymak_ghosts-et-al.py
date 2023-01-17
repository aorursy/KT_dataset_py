import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
#read in the training data and inspect

creature_data_train = pd.read_csv('../input/train.csv')

creature_data_train.head()
#reset the id to be the index

creature_data_train.set_index('id', inplace=True)

creature_data_train.describe()
#would also check the unique colors and creature types:

unique_colors = creature_data_train['color'].unique()

unique_types = creature_data_train['type'].unique()

print(unique_colors, unique_types)
#see how balanced the class labels are...

unique_types_count = creature_data_train['color'].value_counts()

print(unique_types_count)
creature_data_split_train, creature_data_split_test = train_test_split(creature_data_train, test_size = 0.2)
print(len(creature_data_split_train))