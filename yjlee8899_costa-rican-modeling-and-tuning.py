import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import  OneHotEncoder as ohe

from sklearn.preprocessing import StandardScaler as ss

from sklearn.compose import ColumnTransformer as ct

from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier as gbm

from xgboost.sklearn import XGBClassifier

import lightgbm as lgb





from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import f1_score

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics

from xgboost import plot_importance

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix





from bayes_opt import BayesianOptimization

from skopt import BayesSearchCV

from eli5.sklearn import PermutationImportance





%matplotlib inline
# Read in data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.shape, test.shape
train.isnull().values.any()
test.isnull().values.any()
train.info()

train.isnull().values.sum(axis=0)
train_describe = train.describe()

train_describe
test_describe = test.describe()

test_describe
test.isnull().values.sum(axis=0)
plt.figure(figsize=(12, 5))

plt.hist(train.Target.values, bins=4)

plt.title('Histogram - target counts')

plt.xlabel('Count')

plt.ylabel('Target')

plt.show()
plt.title("Distribution of Target")

sns.distplot(train['Target'].dropna(),color='blue', kde=True,bins=100)

plt.show()
sns.set_style("whitegrid")

ax = sns.violinplot(x=train.Target.values)

plt.show()
plt.title("Distribution of log(target)")

sns.distplot(np.log1p(train['Target']).dropna(),color='blue', kde=True,bins=100)

plt.show()
sns.set_style("whitegrid")

ax = sns.violinplot(x=np.log(1+train.Target.values))

plt.show()
yes_no_map = {'no':0,'yes':1}

train['dependency'] = train['dependency'].replace(yes_no_map).astype(np.float32)

train['edjefe'] = train['edjefe'].replace(yes_no_map).astype(np.float32)

train['edjefa'] = train['edjefa'].replace(yes_no_map).astype(np.float32)
train.drop(['Id','idhogar',"dependency","edjefe","edjefa"], inplace = True, axis =1)



test.drop(['Id','idhogar',"dependency","edjefe","edjefa"], inplace = True, axis =1)

y = train.iloc[:,137]

y.unique()
X = train.iloc[:,1:138]

X.shape
my_imputer = SimpleImputer()

X = my_imputer.fit_transform(X)

scale = ss()

X = scale.fit_transform(X)

pca = PCA(0.95)

X = pca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    y,

                                                    test_size = 0.2)
modelrf = rf()
import time

start = time.time()

modelrf = modelrf.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modelrf.predict(X_test)
(classes == y_test).sum()/y_test.size 
KNeighborsClassifier
modelneigh = KNeighborsClassifier(n_neighbors=4)

start = time.time()

modelneigh = modelneigh.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modelneigh.predict(X_test)



classes

(classes == y_test).sum()/y_test.size 
modelgbm=gbm()

start = time.time()

modelgbm = modelgbm.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modelgbm.predict(X_test)



classes

(classes == y_test).sum()/y_test.size 
modellgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',

                             random_state=None, silent=True, metric='None', 

                             n_jobs=4, n_estimators=5000, class_weight='balanced',

                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)
start = time.time()

modellgb = modellgb.fit(X_train, y_train)

end = time.time()

(end-start)/60
classes = modellgb.predict(X_test)



classes

(classes == y_test).sum()/y_test.size 