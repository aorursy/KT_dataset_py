import pandas as pd

import numpy as np
train = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')

test = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')



print(train.shape, test.shape)
train.head()
train.info()
# target variable data

np.unique(train['target'])
# target variable distribution --> imbalanced

1.0 * sum(train['target'])/train.shape[0]
import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
data = []

for f in train.columns:

    # Defining the role

    if f == 'target':

        role = 'target'

    elif f == 'id':

        role = 'id'

    else:

        role = 'input'

        

    # Defining the level

    if 'bin' in f or f == 'target':

        level = 'binary'

    elif 'cat' in f or f == 'id':

        level = 'nominal'

    elif train[f].dtype == float:

        level = 'interval'

    elif train[f].dtype == int:

        level = 'ordinal'

        

    # Initialize keep to True for all variables except for id

    keep = True

    if f == 'id':

        keep = False

    

    # Defining the data type

    dtype = train[f].dtype

    

    # Creating a Dict that contains all the metadata for the variable

    f_dict = {

        'varname': f,

        'role': role,

        'level': level,

        'keep': keep,

        'dtype': dtype

    }

    data.append(f_dict)

    

meta = pd.DataFrame(data, columns = ['varname', 'role', 'level', 'keep', 'dtype'])
meta.head()
binary = meta[(meta.level == 'binary') & (meta.keep)]["varname"].tolist()

category = meta[(meta.level == 'nominal') & (meta.keep)]["varname"].tolist()

integer = meta[(meta.level == 'ordinal') & (meta.keep)]["varname"].tolist()

floats = meta[(meta.level == 'interval') & (meta.keep)]["varname"].tolist()
'''Histogram for Data visualization'''

# target variable into NaN

test['target'] = np.nan



# merge train data with test data

df = pd.concat([train, test], axis=0)

df.head()
def bar_plot(col, data, hue=None):

    f, ax = plt.subplots(figsize=(10, 5))

    sns.countplot(x=col, hue=hue, data=data, alpha=0.5)

    plt.show()

    

def dist_plot(col, data):

    f, ax = plt.subplots(figsize=(10, 5))

    sns.distplot(data[col].dropna(), kde=False, bins=10)

    plt.show()

    

for col in binary + category + integer:

    bar_plot(col, df)

    

for col in floats:

    dist_plot(col, df)
def corr_heatmap(v):

    correlations = train[v].corr()

    

    # Create color map ranging between two colors

    cmap = sns.diverging_palette(220, 10, as_cmap = True)

    

    fig, ax = plt.subplots(figsize = (10, 7))

    sns.heatmap(correlations, cmap = cmap, vmax = 1.0, center = 0, fmt = '.2f',

                square = True, linewidth = .5, annot = True, 

                cbar_kws = {"shrink" : .75})

    plt.show();

    

meta.set_index('varname', inplace = True)

v = meta[(meta.level == 'interval') & (meta.keep)].index

corr_heatmap(v)
def bar_plot_ci(col, data):

    f, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(x=col, y='target', data=data)

    plt.show()

    

for col in binary + category + integer:

    bar_plot_ci(col, df)
# df only for test data

df['is_test'] = df['target'].isnull()



for col in binary + category + integer:

    bar_plot(col, df, 'is_test')
train_label = train['target']

train_id = train['id']

del train['target'], train['id']



test_id = test['id']

del test['id']
# Derived 01: count -1(NaN)

train['missing'] = (train == -1).sum(axis = 1).astype(float)

test['missing'] = (test == -1).sum(axis = 1).astype(float)



# Derived 02: sum of binary variables

bin_features = [c for c in train.columns if 'bin' in c]

train['bin_sum'] = train[bin_features].sum(axis = 1)

test['bin_sum'] = test[bin_features].sum(axis = 1)



# Derived 03: target encoding (features are selected based on the target rate above)

features = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 

            'ps_ind_12_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin',

            'ps_ind_04_cat', 'ps_ind_05_cat', 

            'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat',

            'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_11_cat',

            'ps_ind_01', 'ps_ind_03', 'ps_ind_15', 'ps_car_11']