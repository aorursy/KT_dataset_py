import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

%matplotlib inline
train = pd.read_csv('../input/train.csv', index_col='ID')
test = pd.read_csv('../input/test.csv', index_col='ID')

train.head()
print('\trows\tcolumns')
print('Train:\t{:>6,}\t{:>6,}'.format(*train.shape))
print('Test:\t{:>6,}\t{:>6,}'.format(*test.shape))
target = train.pop('target')

sns.set()
mn_train = train.mean()
std_train = train.std()

mn_test = test.mean()
std_test = test.std()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax = sns.distplot(mn_train, kde=False, norm_hist=False, ax=ax)
sns.distplot(mn_test, kde=False, norm_hist=False, ax=ax)
ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_title('Distribution of the mean value of train/test features.')
ax.set_xlabel(r'Mean value ($\mu$)')
ax.set_ylabel('Number of features')
ax.legend(['train', 'test'])

ax = axes[1]
sns.distplot(std_train, kde=False, norm_hist=False, ax=ax)
sns.distplot(std_test, kde=False, norm_hist=False, ax=ax)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_title('Distribution of std value of train/test features.')
ax.set_xlabel(r'Standard Deviation ($\sigma^2$)')
ax.set_ylabel('Number of features')
ax.legend(['Train', 'Test']);
cr = (train.max() == 0) & (train.min() == 0)
train_all_zero_feature_count = cr.index[cr].shape[0]
cr2 = (test.max() == 0) & (test.max() == 0)
test_all_zero_feature_count = cr2.index[cr2].shape[0]
print('Number of training features with all 0 values:\t{}'.format(train_all_zero_feature_count))
print('Number of test features with all 0 values:\t{}'.format(test_all_zero_feature_count))
train_all_zero_features = cr.index[cr]
train.drop(columns=train_all_zero_features, inplace=True)

count_of_binary_features = (train.max() == 1).sum()
print('Number of binary features: {}'.format(count_of_binary_features))
less_than_1000_count = (train.max() < 1000).sum()
print('Number of train features with max value < 1,000: {}'.format(less_than_1000_count))
plt.figure(figsize=(13, 5))
ax = sns.distplot(train.max(), kde=False, norm_hist=False, bins=1000)
ax = sns.distplot(test.max(), kde=False, norm_hist=False, bins=1000)
plt.xlim(left=-20000000, right=1e9)
ax.set_xlabel('Max feature value (axis clipped @ 1e9)')
ax.set_ylabel('Number of features')
ax.set_title('Distribution of max value of features')
ax.legend(['Train', 'Test']);
plt.figure(figsize=(13, 5))
ax = sns.distplot(target, kde=False, norm_hist=False, bins=200)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_xlabel('Transaction value')
ax.set_ylabel('Number of transactions')
ax.set_title('Distribution of target transaction values');