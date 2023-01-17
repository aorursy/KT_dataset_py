import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



import seaborn as sns # visualization

import missingno as msno



from sklearn.model_selection import train_test_split

# from sklearn.cluster import MiniBatchKMeans

from sklearn.model_selection import StratifiedShuffleSplit



import xgboost as xgb # Gradient Boosting

from xgboost import XGBClassifier

import warnings



warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output
np.random.seed(1989)

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("Train shape : ", train.shape)

print("Test shape : ", test.shape )
train.head()
train.info()
test.info()
print("Train has these types: {}".format(train.dtypes.unique()))

print("Test has these types: {}".format(test.dtypes.unique()))
targets = train['target'].values

# sns.set(style="darkgrid")

ax = sns.countplot(x = targets)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x()+ 0.3, p.get_height()+10000))

plt.title('Distribution of Target', fontsize=20)

plt.xlabel('Claim', fontsize=20)

plt.ylabel('Frequency [%]', fontsize=20)

ax.set_ylim(top=700000)
print('Id is unique.') if train.id.nunique() == train.shape[0] else print('Oh no')

print('Train and test sets are distinct.') if len(np.intersect1d(train.id.values, test.id.values)) == 0 else print('Oh no')

print('We do not need to worry about missing values.') if train.count().min() == train.shape[0] else print('Oh no')
train_nan = train

train_nan = train_nan.replace(-1, np.NaN)



msno.matrix(df=train_nan.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))   
test_nan = test

test_nan = test_nan.replace(-1, np.NaN)



msno.matrix(df=test_nan.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))   
# Extract columns with null data

train_nan = train_nan.loc[:, train_nan.isnull().any()]

train_nan_columns = train_nan.columns



test_nan = test_nan.loc[:, test_nan.isnull().any()]

test_nan_columns = test_nan.columns
print('Columns \t Number of NaN')

for column in train_nan.columns:

    print('{}:\t {}'.format(column,len(train_nan[column][np.isnan(train_nan[column])])))
print('Columns \t Number of NaN')

for column in test_nan.columns:

    print('{}:\t {}'.format(column,len(test_nan[column][np.isnan(test_nan[column])])))
feature_list = list(train.columns)

def groupFeatures(features):

    features_bin = []

    features_cat = []

    features_etc = []

    for feature in features:

        if 'bin' in feature and 'calc' not in feature:

            features_bin.append(feature)

        elif 'cat' in feature:

            features_cat.append(feature)

        elif 'id' in feature or 'target' in feature:

            continue

        else:

            features_etc.append(feature)

    return features_bin, features_cat, features_etc



feature_list_bin, feature_list_cat, feature_list_etc = groupFeatures(feature_list)

print("# of binary feature : ", len(feature_list_bin))

print("# of categorical feature : ", len(feature_list_cat))

print("# of other feature : ", len(feature_list_etc))
colormap = "jet"

plt.figure(figsize=(16, 12))

plt.title('Pearson correlation of continuous features', y=1.05, size=15)

data = train.drop(['id'], axis=1)

sns.heatmap(data.corr(),linewidths=0.1,vmax=1.0, vmin=-1.0, square=True, cmap=colormap, linecolor='white')
feature_list_calc = []

feature_list_without_calc = []

for feature in train.columns:

    if 'calc' in feature or 'target' in feature:

        feature_list_calc.append(feature)

    else:

        feature_list_without_calc.append(feature)

        

train_without_calc = train.drop(feature_list_calc, axis=1).drop(['id'], axis=1)

train_with_calc = train.drop(feature_list_without_calc, axis=1)
colormap = "jet"

plt.figure(figsize=(20, 20))

plt.title('Pearson correlation of continuous features', y=1.05, size=15)

sns.heatmap(train_with_calc.corr(),linewidths=0.1,

            vmax=1.0, vmin=-1.0, square=True, cmap=colormap, linecolor='white')
colormap = "jet"

plt.figure(figsize=(20, 20))

plt.title('Pearson correlation of continuous features', y=1.05, size=15)

sns.heatmap(train_without_calc.corr(),linewidths=0.1,

            vmax=1.0, vmin=-1.0, square=True, cmap=colormap, linecolor='white')
def TrainTestHistogram(train, test, feature):

    fig, axes = plt.subplots(len(feature), 2, figsize=(10, 40))

    fig.tight_layout()



    left  = 0  # the left side of the subplots of the figure

    right = 0.9    # the right side of the subplots of the figure

    bottom = 0.1   # the bottom of the subplots of the figure

    top = 0.9      # the top of the subplots of the figure

    wspace = 0.3   # the amount of width reserved for blank space between subplots

    hspace = 0.7   # the amount of height reserved for white space between subplot



    plt.subplots_adjust(left=left, bottom=bottom, right=right, 

                        top=top, wspace=wspace, hspace=hspace)

    count = 0

    for i, ax in enumerate(axes.ravel()):

        if i % 2 == 0:

            title = 'Train: ' + feature[count]

            ax.hist(train[feature[count]], bins=30, normed=False)

            ax.set_title(title)

#             ax.text(0, 1.2, train[feature[count]].head(), horizontalalignment='left',

#                     verticalalignment='top', style='italic',

#                 bbox={'facecolor':'red', 'alpha':0.2, 'pad':10}, transform=ax.transAxes)

        else:

            title = 'Test: ' + feature[count]

            ax.hist(test[feature[count]], bins=30, normed=False)

            ax.set_title(title)

#             ax.text(0, 1.2, test[feature[count]].head(), horizontalalignment='left',

#                     verticalalignment='top', style='italic',

#                 bbox={'facecolor':'red', 'alpha':0.2, 'pad':10}, transform=ax.transAxes)

            count = count + 1
TrainTestHistogram(train, test, feature_list_bin)
TrainTestHistogram(train, test, feature_list_cat)
TrainTestHistogram(train, test, feature_list_etc)
left  = 0  # the left side of the subplots of the figure

right = 0.9    # the right side of the subplots of the figure

bottom = 0.1   # the bottom of the subplots of the figure

top = 0.9      # the top of the subplots of the figure

wspace = 0.3   # the amount of width reserved for blank space between subplots

hspace = 0.7   # the amount of height reserved for white space between subplot



fig, axes = plt.subplots(13, 2, figsize=(10, 40))

plt.subplots_adjust(left=left, bottom=bottom, right=right, 

                    top=top, wspace=wspace, hspace=hspace)



for i, ax in enumerate(axes.ravel()):

    title = 'Train: ' + feature_list_etc[i]

    ax.hist(train[feature_list_etc[i]], bins=20, normed=True)

    ax.set_title(title)

    ax.text(0, 1.2, train[feature_list_etc[i]].head(), horizontalalignment='left',

            verticalalignment='top', style='italic',

       bbox={'facecolor':'red', 'alpha':0.2, 'pad':10}, transform=ax.transAxes)
# For ordinal group

ordianal_features = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_reg_01',

                    'ps_reg_02', 'ps_car_11', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03',

                    'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08',

                    'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13',

                    'ps_calc_14']



continuous_features = ['ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15']
ordinal_feature_with_calc = [feature for feature in ordianal_features if 'calc' in feature]

ordinal_feature_without_calc = [feature for feature in ordianal_features if 'calc' not in feature]
sns.set(font_scale=1.5)

for i in range(len(feature_list_cat)):

    feature_number = i

    temp_data = train.loc[train[feature_list_cat[feature_number]] != -1]

    g = sns.factorplot(x=feature_list_cat[feature_number], y="target", data=temp_data, kind="bar",

                   size=6, palette = "muted")

    g = g.set_ylabels("Claim probability")
sns.set(font_scale=1.5)

for i in range(len(feature_list_bin)):

    feature_number = i

    temp_data = train.loc[train[feature_list_bin[feature_number]] != -1]

    g = sns.factorplot(x=feature_list_bin[feature_number], y="target", data=temp_data, kind="bar",

                   size=6, palette = "muted")

    g = g.set_ylabels("Claim probability")
sns.set(font_scale=1.5)

for i in range(len(ordinal_feature_with_calc)):

    feature_number = i

    temp_data = train.loc[train[ordinal_feature_with_calc[feature_number]] != -1]

    g = sns.factorplot(x=ordinal_feature_with_calc[feature_number], y="target", data=temp_data, kind="bar",

                   size=6, palette = "muted")

    g = g.set_ylabels("Claim probability")
sns.set(font_scale=1.5)

for i in range(len(ordinal_feature_without_calc)):

    feature_number = i

    temp_data = train.loc[train[ordinal_feature_without_calc[feature_number]] != -1]

    g = sns.factorplot(x=ordinal_feature_without_calc[feature_number], y="target", data=temp_data, kind="bar",

                   size=6, palette = "muted")

    g = g.set_ylabels("Claim probability")
for feature in continuous_features:

    g = sns.FacetGrid(train, col='target')

    g = g.map(sns.distplot, feature)
def make_overlap_histogram(df, feature, target):

    fig, ax = plt.subplots(figsize=(6, 6))

    g = sns.kdeplot(df[feature][(df[target] == 0) & (df[feature] != -1)], color="Red", shade = True)

    g = sns.kdeplot(df[feature][(df[target] == 1) & (df[feature] != -1)], ax=g, color="Blue", shade= True)

    g.set_xlabel(feature)

    g.set_ylabel("Frequency")

    g = g.legend(["Not claim","Claim"])
for feature in continuous_features:

    make_overlap_histogram(train, feature, 'target')
def show_skew(df, feature):

    fig, ax = plt.subplots(figsize=(8, 8))

    g = sns.distplot(train[feature][train[feature] != -1], color="m", label="Skewness : %.2f"%(train[feature].skew()))

    g = g.legend(loc="best")
dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

dataset = dataset.replace(-1, np.NaN)

dataset.tail()
for feature in continuous_features:

    show_skew(dataset, feature)
for feature in continuous_features:

    fig, ax = plt.subplots(figsize=(6, 6))

    temp = dataset[feature].map(lambda i: np.log(i) if i > 0 else 0)

    g = sns.distplot(temp[temp != -1], color="m", label="Skewness : %.2f"%(temp.skew()))

    g = g.legend(loc="best")
feature_list_not_using = ['ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_08', 'ps_calc_09']

feature_list_log_function = ['ps_car_13']
dataset = dataset.drop(feature_list_not_using, axis=1)
dataset['ps_car_13'] = dataset['ps_car_13'].map(lambda i: np.log(i) + 1 if i > 0 else 0)
for feature in (set(train_nan_columns).union(set(test_nan_columns))):

    if 'cat' in feature or 'bin' in feature:

        # For categorical and binary features with postfix, substitue null values with the most frequent value to avoid float number.

        dataset[feature].fillna(dataset[feature].value_counts().idxmax(), inplace=True)

    elif feature in continuous_features:

        dataset[feature].fillna(dataset[feature].median(), inplace=True)

    elif feature in ordianal_features:

        # For ordinal features which was assumed, substitue null values with the most frequent value to avoid float number.

        dataset[feature].fillna(dataset[feature].value_counts().idxmax(), inplace=True)

    else:

        print(feature)
msno.matrix(df=dataset.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))  
for feature in feature_list_cat:

    print("{}: \t{}".format(feature, dataset[feature].value_counts().shape[0]))
def oneHotEncode_dataframe(df, features):

    for feature in features:

        if df[feature].value_counts().shape[0] < 8:

            temp_onehot_encoded = pd.get_dummies(df[feature])

            column_names = ["{}_{}".format(feature, x) for x in temp_onehot_encoded.columns]

            temp_onehot_encoded.columns = column_names

            df = df.drop(feature, axis=1)

            df = pd.concat([df, temp_onehot_encoded], axis=1)

        else:

            continue

    return df
dataset = oneHotEncode_dataframe(dataset, feature_list_cat)
x_train = dataset.loc[:train.shape[0]-1, :]

x_test = dataset.loc[train.shape[0]:, :]
x_train.tail()
x_test.head()
# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897

def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)

 

def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score