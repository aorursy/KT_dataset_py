import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('../input/train.csv')
print(train.size)
print(train.dtypes)
print(train.groupby(['landmark_id']).landmark_id.count())
# Occurance of landmark_id in decreasing order(Top categories)
temp = pd.DataFrame(train.landmark_id.value_counts().head(50))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id','count']
temp