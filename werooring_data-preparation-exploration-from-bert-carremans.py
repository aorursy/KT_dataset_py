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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectFromModel

from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier



pd.set_option('display.max_columns', 100)
debug = False

if debug:

    NROWS = 50000

else:

    NROWS = None
train = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv', nrows=NROWS)

test = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv', nrows=NROWS)
train.head()
train.shape
train.drop_duplicates()

train.shape
test.shape
train.info()
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

    elif train[f].dtype == 'float64':

        level = 'interval'

    elif train[f].dtype == 'int64':

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

    

meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])

meta.set_index('varname', inplace=True)
meta.head()
meta.loc[(meta.level == 'nominal') & (meta.keep)].index
pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()
meta.groupby(['role', 'level'])['role'].size()
interval_vars = meta.loc[(meta.level == 'interval') & (meta.keep)].index

train[interval_vars].describe()
ordinal_vars = meta.loc[(meta.level == 'ordinal') & (meta.keep)].index

train[ordinal_vars].describe()
binary_vars = meta[(meta.level == 'binary') & (meta.keep)].index

train[binary_vars].describe()
desired_apriori=0.1



# Get the indices per target value

idx_0 = train.loc[train.target == 0].index

idx_1 = train.loc[train.target == 1].index



# Get original number of records per target value

num_0 = len(train.loc[idx_0])

num_1 = len(train.loc[idx_1])



# Calculate the undersampling rate and resulting number of records with target=0

undersampling_rate = ((1-desired_apriori)*num_1)/(num_0*desired_apriori)

undersampled_num_0 = int(undersampling_rate*num_0)

print('Rate to undersample records with target=0: {}'.format(undersampling_rate))

print('Number of records with target=0 after undersampling: {}'.format(undersampled_num_0))



# Randomly select records with target=0 to get at the desired a priori

undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_num_0)



# Construct list with remaining indices

idx_list = list(undersampled_idx) + list(idx_1)



# Return undersample data frame

train = train.loc[idx_list].reset_index(drop=True)
vars_with_missing = []



for col in train.columns:

    num_missings = train[train[col] == -1][col].count()

    if num_missings > 0:

        vars_with_missing.append(col)

        missings_percent = num_missings/train.shape[0]

        

        print('Variable {} has {} records ({:.2%}) with missing values'.format(col, num_missings, missings_percent))

        

print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))
# Dropping the variables with too many missing values

vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']

train.drop(vars_to_drop, axis=1, inplace=True)

meta.loc[(vars_to_drop), 'keep'] = False # Updating the meta
# Imputing with the mean or mode

mean_imputer = SimpleImputer(missing_values=-1, strategy='mean')

mode_imputer = SimpleImputer(missing_values=-1, strategy='most_frequent')

train['ps_reg_03'] = mean_imputer.fit_transform(train[['ps_reg_03']]).ravel()

train['ps_car_11'] = mode_imputer.fit_transform(train[['ps_car_11']]).ravel()

train['ps_car_14'] = mean_imputer.fit_transform(train[['ps_car_14']]).ravel()
nominal_vars = meta.loc[(meta.level == 'nominal') & (meta.keep)].index

for col in nominal_vars:

    num_dist_values = train[col].nunique()

    print('Variable {} has {} disticnt values'.format(col, num_dist_values))
def add_noise(series, noise_level):

    return series * (1 + noise_level * np.random.randn(len(series)))



def target_encode(trn_series=None,

                 tst_series=None,

                 target=None,

                 min_samples_leaf=1,

                 smoothing=1,

                 noise_level=0):

    """

    trn_series : training categorical feature as a pd.Series

    tst_series : test categorical feature as a pd.Series

    target : target data as a pd.Series

    min_samples_leaf (int) : minimum samples to take category average into account

    smoothing (int) : smoothing effect to balance categorical average vs prior  

    """

    assert len(trn_series) == len(target)

    assert trn_series.name == tst_series.name

    temp = pd.concat([trn_series, target], axis=1)

    

    # Compute target mean

    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    

    # Compute smoothing

    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    

    # Apply average function to all target data

    prior = target.mean()

    

    # The bigger the count the less full_avg is taken into account

    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing

    averages.drop(["mean", "count"], axis=1, inplace=True)

    

    # Apply averages to trn ad tst series

    ft_trn_series = pd.merge(

        trn_series.to_frame(trn_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=trn_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)



    # pd.merge does not keep the index to restore it

    ft_trn_series.index = trn_series.index

    ft_tst_series = pd.merge(

        tst_series.to_frame(tst_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=tst_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    

    # pd.merge does not keep the index so restore it

    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
train_encoded, test_encoded = target_encode(train["ps_car_11_cat"], 

                             test["ps_car_11_cat"], 

                             target=train.target, 

                             min_samples_leaf=100,

                             smoothing=10,

                             noise_level=0.01)
train_encoded
test_encoded
train['ps_car_11_cat_te'] = train_encoded

train.drop('ps_car_11_cat', axis=1, inplace=True)

meta.loc['ps_car_11_cat','keep'] = False  # Updating the meta

test['ps_car_11_cat_te'] = test_encoded

test.drop('ps_car_11_cat', axis=1, inplace=True)
train.ps_car_11_cat_te.nunique()
sns.set(font_scale=1.2)

nominal_vars = meta.loc[(meta.level == 'nominal') & (meta.keep)].index



for col in nominal_vars:

    plt.figure()

    fig, ax = plt.subplots(figsize=(8, 4))

    # Calculate the precentage of target=1 per category value

    cat_perc = train[[col, 'target']].groupby([col], as_index=False).mean()

    cat_perc.sort_values(by='target', ascending=False, inplace=True)

    # Bar plot

    # Order the bars descending on target mean

    sns.barplot(x=col, y='target', data=cat_perc, order=cat_perc[col], ax=ax)

    plt.ylabel('% target')

    plt.xlabel(col)

    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.show();    
def corr_heatmap(interval_vars):

    correlations = train[interval_vars].corr()

    

    # Create color map ranging between two colors

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(10,10))

    sns.heatmap(correlations, cmap=cmap, vmax=1, center=0, fmt='.2f',

               square=True, linewidth=.5, annot=True,

               cbar_kws={'shrink': 0.75})

    plt.show();



interval_vars = meta.loc[(meta.level=='interval') & (meta.keep)].index

corr_heatmap(interval_vars)
train_sample = train.sample(frac=0.1)
sns.lmplot(x='ps_reg_02', y='ps_reg_03', data=train_sample, hue='target',

          palette='Set1', scatter_kws={'alpha': 0.3})

plt.show()
sns.lmplot(x='ps_car_12', y='ps_car_13', data=train_sample, hue='target',

          palette='Set1', scatter_kws={'alpha': 0.3})

plt.show()
sns.lmplot(x='ps_car_12', y='ps_car_14', data=train_sample, hue='target',

          palette='Set1', scatter_kws={'alpha': 0.3})

plt.show()
sns.lmplot(x='ps_car_13', y='ps_car_15', data=train_sample, hue='target',

          palette='Set1', scatter_kws={'alpha': 0.3})

plt.show()
sns.set(font_scale=1)

oridnal_vars = meta.loc[(meta.level=='ordinal') & (meta.keep)].index

corr_heatmap(ordinal_vars)
train.head()
print('Before dummification we have {} variables in train.'.format(train.shape[1]))

train = pd.get_dummies(train, columns=nominal_vars, drop_first=True)

print('After dummification we have {} variables in train'.format(train.shape[1]))
train.head()
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

interactions = pd.DataFrame(data=poly.fit_transform(train[interval_vars]),

                           columns=poly.get_feature_names(interval_vars))

interactions.drop(interval_vars, axis=1, inplace=True) # Remove the original columns

# Concat the interaction variables to the train data

print('Before creating intercations we have {} variables in train.'.format(train.shape[1]))

train = pd.concat([train, interactions], axis=1)

print('After creating intercations we have {} variables in train.'.format(train.shape[1]))
selector = VarianceThreshold(threshold=0.01)

selector.fit(train.drop(['id', 'target'],axis=1)) # Fit to train without id and target variables
selector.get_support()
vfunc = np.vectorize(lambda x: not x) # Function to toggle boolean array elements

not_selected_columns = train.drop(['id', 'target'], axis=1).columns[vfunc(selector.get_support())]

print('{} variables have too low variance'.format(len(not_selected_columns)))

print('These variables are {}'.format(list(not_selected_columns)))
X_train = train.drop(['id', 'target'], axis=1)

y_train = train['target']



feat_labels = X_train.columns



rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

rf.fit(X_train, y_train)

importances = rf.feature_importances_



indices = np.argsort(rf.feature_importances_)[::-1]
np.argsort(rf.feature_importances_)
indices
for f in range(X_train.shape[1]):

    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], 

                            importances[indices[f]]))
sfm = SelectFromModel(rf, threshold='median', prefit=True)

print('Number of features before a selection: {}'.format(X_train.shape[1]))

n_features = sfm.transform(X_train).shape[1]

print('Number of features after selection: {}'.format(n_features))

selected_vars = list(feat_labels[sfm.get_support()])
train = train[selected_vars + ['target']]
scaler = StandardScaler()

scaler.fit_transform(train.drop(['target'], axis=1))