# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn import svm, neighbors, linear_model, neural_network

from sklearn.svm import NuSVC

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from tqdm import tqdm_notebook

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



import warnings

warnings.filterwarnings('ignore')
print("# Loading data...") 

train = pd.read_csv('../input/train.csv', header=0)

test = pd.read_csv('../input/test.csv', header=0)
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]



data = pd.concat([pd.DataFrame(train[cols]), pd.DataFrame(test[cols])])

data2 = StandardScaler().fit_transform(VarianceThreshold(threshold=1.5).fit_transform(data[cols]))

data_magic=pd.concat([pd.DataFrame(train['wheezy-copper-turtle-magic']), pd.DataFrame(test['wheezy-copper-turtle-magic'])])

data_770= pd.DataFrame(data2)

data_770.reset_index(drop=True,inplace=True)

data_magic.reset_index(drop=True,inplace=True)
data_770=pd.concat([data_magic,data_770],axis=1)

data_770.head()
#### ONE-HOT-ENCODE THE MAGIC FEATURE

len_train = train.shape[0]

test['target'] = -1

data_new = pd.concat([data_770, pd.get_dummies(data_770['wheezy-copper-turtle-magic'])], axis=1, sort=False)



train_one = data_new.iloc[:len_train,:]

test_one = data_new.iloc[len_train:,:]
train_one=train_one.drop(['wheezy-copper-turtle-magic'],axis=1)

test_one=test_one.drop(['wheezy-copper-turtle-magic'],axis=1)
train_one.head()
test_one.head()
train_one.to_csv("train_var_onehot.csv")

test_one.to_csv("test_var_onehot.csv")
print("over")