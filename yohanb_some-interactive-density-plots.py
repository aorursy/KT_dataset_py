import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import holoviews as hv

from holoviews import opts, dim, Palette

hv.extension('bokeh')



### Load the data

input_path = '../input/'

df_train = pd.read_csv(input_path + 'train.csv')

df_test = pd.read_csv(input_path + 'test.csv')

### Feature names (removing 'ID' and 'Magic' one)

features = [c for c in df_test.columns if c not in ['id', 'wheezy-copper-turtle-magic']]
a_feature = 20
n = 100 # number of features. The kernel can't render a picture for each of 512 values what's why we limit that number to 100. 

# If you want to see distributions for other values use `range(m,n)` (theoretical it should be less than 500, but actually it would be better to use 100 or lees at a time).



### Filtering only rows that has values from 0 to 99 in 'wheezy-copper-turtle-magic':

df_n = df_train[df_train['wheezy-copper-turtle-magic'].isin(range(n))]



# Separating targent 0 and 1:

df_n_0 = df_n[df_n['target'] == 0]

df_n_1 = df_n[df_n['target'] == 1]





### Forming datasets

exp = hv.Dataset(df_train[[features[a_feature], 'wheezy-copper-turtle-magic']], [features[a_feature]])

exp_n = hv.Dataset(df_n[[features[a_feature], 'wheezy-copper-turtle-magic']], [features[a_feature]])

exp_0 = hv.Dataset(df_n_0[[features[a_feature], 'wheezy-copper-turtle-magic']], [features[a_feature]])

exp_1 = hv.Dataset(df_n_1[[features[a_feature], 'wheezy-copper-turtle-magic']], [features[a_feature]])



### Pictures

# Density plot for `a_feature` (all values of `wheezy-copper-turtle-magic`)

p1 = exp.to(hv.Distribution, features[a_feature])

# Density plot for `a_feature` (for a single value of `wheezy-copper-turtle-magic`)

p2 = exp_n.to(hv.Distribution, features[a_feature], groupby='wheezy-copper-turtle-magic', label = 'Target combined') 

# Density plot for `a_feature` (for a single value of `wheezy-copper-turtle-magic`fot targets 0 and 1 separately):

p3 = exp_0.to(hv.Distribution, features[a_feature], groupby='wheezy-copper-turtle-magic', label = '0') 

p4 = exp_1.to(hv.Distribution, features[a_feature], groupby='wheezy-copper-turtle-magic', label = '1')

### Show the pictures:

layout = hv.Layout((p1) + (p2) + (p3 * p4)).cols(1)

layout