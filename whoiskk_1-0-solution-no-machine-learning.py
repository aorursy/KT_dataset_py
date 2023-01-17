import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



plt.rcParams['figure.figsize'] = [9, 12]



import warnings

warnings.simplefilter('ignore')
train = pd.read_csv("/kaggle/input/whoisafriend/train.csv")

test = pd.read_csv("/kaggle/input/whoisafriend/test.csv")

sub = pd.read_csv("/kaggle/input/whoisafriend/sample_submission.csv")



train.shape, test.shape, sub.shape
agg_train = train.groupby(['Person A', 'Person B'])['Years of Knowing'].count().reset_index()

agg_train.rename({

    "Years of Knowing": "Interaction Count"

}, axis=1, inplace=True)



agg_test = test.groupby(['Person A', 'Person B'])['Years of Knowing'].count().reset_index()

agg_test.rename({

    "Years of Knowing": "Interaction Count"

}, axis=1, inplace=True)
agg_train.head()
train = pd.merge(train, agg_train, on=['Person A', 'Person B'], how='left')

test = pd.merge(test, agg_test, on=['Person A', 'Person B'], how='left')
sns.lmplot('Interaction Count', 'Friends', data=train, fit_reg=False)
plt.figure(figsize=(12, 5))

sns.countplot('Interaction Count', data=train, hue='Friends')

plt.show()
test['Friends'] = np.nan

test['Friends'] = test['Interaction Count'].apply(lambda x: 1 if x > 7 else 0)
test[['ID', 'Friends']].to_csv("1.0_sub.csv", index=False)