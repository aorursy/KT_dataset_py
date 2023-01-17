import os 

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

%matplotlib inline

%config InlineBackend.figure_format = 'svg'

import seaborn as sns
BASE_DIR = '../input/lish-moa/'

train_features = pd.read_csv(BASE_DIR + 'train_features.csv')

train_targets_scored = pd.read_csv(BASE_DIR + 'train_targets_scored.csv')

train_targets_nonscored = pd.read_csv(BASE_DIR + 'train_targets_nonscored.csv')



test_features = pd.read_csv(BASE_DIR + 'test_features.csv')

sample_submission = pd.read_csv(BASE_DIR + 'sample_submission.csv')



# TRAIN FEATURES

INDEX = 'sig_id'

g_cols = [col for col in train_features.columns if col.startswith('g-')]

c_cols = [col for col in train_features.columns if col.startswith('c-')]



other_cols = ['cp_type', 'cp_time', 'cp_dose']  # Categoricals
train_features.head()
print("Q: Does the features dataframe have any null entries?")

if not train_features.isnull().values.any():

    print('A: Nope, none!')
print('A full list of categorical features')

print('----'*10)

for col in other_cols:

    print('Number of unique values in "%s": %d' % (col, train_features[col].nunique()))

    print('Values: ', train_features[col].unique())

    print('')
fig, ax = plt.subplots(1, 3, figsize=(10, 3))

sns.countplot(data=train_features, x=other_cols[0], ax=ax[0])

sns.countplot(data=train_features, x=other_cols[1], ax=ax[1])

sns.countplot(data=train_features, x=other_cols[2], ax=ax[2])

plt.tight_layout()

fig.suptitle('Distribution of the categorical variables', y=1.1)

plt.show()
print('Q: Does the features dataframe have any duplicated rows?')

if not train_features.duplicated().values.any():

    print('A: Nope!')
print('Q: Are there any duplicated rows associated with different sig_ids?')

if not train_features.loc[:, train_features.columns !=INDEX].duplicated().values.any():

    print('A: Nope!')
print('Feature set includes a series of c- and g- columns.')

print('Number of c_cols: %d' % (len(c_cols)))

print('Number of g_cols: %d' % (len(g_cols)))
def display_distributions(cols):

    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(12, 10))

    for i in range(len(cols)):

        # print(i)

        sns.distplot(train_features[cols[i]], ax = axs[i // 5, i % 5], norm_hist=False, kde=False)

        #axs[i // 10, i % 10].set_title(le.inverse_transform(np.argmax([samples[0][1][i].numpy()], axis=-1))[0])

    plt.tight_layout() # w_pad=0.01, h_pad=1

    plt.show()



display_distributions(c_cols[:25])

display_distributions(c_cols[25:50])

display_distributions(c_cols[50:75])

display_distributions(c_cols[75:100])
display_distributions(g_cols[:25])

display_distributions(g_cols[25:50])

display_distributions(g_cols[50:75])

display_distributions(g_cols[75:100])
train_targets_scored.head()
target_cols = train_targets_scored.columns[1:]  # 1 removes the sig_id

print('Number of target columns: %d' % (len(target_cols)))

print('We are predicting 206 columns for each sig_id')
total = 0

for i in target_cols:

    total += train_targets_scored[i].nunique()



if total/(len(target_cols)) == 2:

    print('All of the target columns are binary.')
train_targets_scored.set_index('sig_id', inplace=True)
target_freq = train_targets_scored.sum(axis=0).to_frame('Counts').sort_values('Counts').reset_index()



fig, ax = plt.subplots(1, 2, figsize=(8, 4))

sns.barplot(target_freq.index, target_freq.Counts, ax=ax[0]).set(xticklabels=[])

ax[0].set_xlabel('Classes')



ax[1] = sns.distplot(target_freq.Counts, kde=False)

ax[1].set_ylabel('Counts of Counts')

fig.suptitle('Class Distribution', y=1.1)

plt.tight_layout()

plt.show()
target_freq[:5]
target_freq[-5:]