# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.preprocessing import LabelBinarizer , LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_selection import SelectKBest , chi2 , f_classif

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss , hamming_loss

from skmultilearn.adapt import MLkNN

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_unscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')

sample_sub = pd.read_csv('../input/lish-moa/sample_submission.csv')
def EncodeLabel(data , feature , binary=True):

    if binary:

        lb = LabelBinarizer()

        temp = lb.fit_transform(data[feature])

        data[feature]= temp

    else:

        le = LabelEncoder()

        temp = le.fit_transform(data[feature])

        data[feature] = temp
print(len(train_targets_scored.columns))

print(train_targets_scored.columns)

for c in train_targets_scored.columns[1:]:

    print(train_targets_scored[c].value_counts())
train_features.describe()
train_features.isnull().sum().sum()
print(f'total number of samples = {train_features.shape[0]}')

print(f'total number of features = {len(train_features.columns[1:])}')

gene_exp_features = [c for c in train_features.columns if c.startswith('g-')]

print(f'total number of gene expression features {gene_exp_features[0]} to {gene_exp_features[-1]} = {len(gene_exp_features)}')

cell_viability = [c for c in train_features.columns if c.startswith('c-')]

print(f'total number of cell viability features {cell_viability[0]} to {cell_viability[-1]} = {len(cell_viability)}')

other_features = [c for c in train_features.columns[1:] if c not in gene_exp_features and c not in cell_viability]

print('other features')

for c in other_features:

    print(c)
print(f'dtype of g- = {train_features[gene_exp_features[0]].dtypes}')

print(f'dtype of c- = {train_features[cell_viability[0]].dtypes}')

print(f'dtype of {other_features[0]}={train_features[other_features[0]].dtypes}')

print(f'dtype of {other_features[1]}={train_features[other_features[1]].dtypes}')

print(f'dtype of {other_features[2]}={train_features[other_features[2]].dtypes}')
for c in other_features:

    print(f'no. of unique values for {c} = {train_features[c].nunique()}')
EncodeLabel(train_features , 'cp_type')

EncodeLabel(train_features , 'cp_dose')
X = train_features.drop(columns=['sig_id'])

Y = train_targets_scored.drop(columns=['sig_id'])
train_x , val_x , train_y , val_y = train_test_split(X,Y, test_size=0.2)

print(f'shape of train_x = {train_x.shape}')

print(f'shape of train_y = {train_y.shape}')

print(f'shape of val_x = {val_x.shape}')

print(f'shape of val_y = {val_y.shape}')
classifier = MLkNN(k=3)

classifier.fit(np.array(train_x) ,np.array( train_y))

preds = classifier.predict(np.array(val_x))

loss = hamming_loss(val_y , preds)

print(loss)
EncodeLabel(test_features , 'cp_type')

EncodeLabel(test_features , 'cp_dose')
x_test = np.array(test_features.drop(columns=['sig_id']))

y_test = classifier.predict(x_test)
y_dense = y_test.todense()

print(y_dense.shape)

print(y_test.shape)

y_dense
pred_df = test_features[['sig_id']]

for i, d in enumerate(val_y.columns):

    pred_df[d] = y_dense[:,i]

    

pred_df.head()
pred_df.set_index('sig_id' , inplace=True)

pred_df
pred_df.to_csv('submission.csv')