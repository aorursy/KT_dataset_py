# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

%matplotlib inline 

import warnings
warnings.filterwarnings('ignore')


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder

sns.set_style('whitegrid')
pd.set_option('display.max_columns', None) # display all columns

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv("../input/sample_submission.csv")
### Exploration
test.head()

### Drop "OutcomeSubtype" feature, not used in test set
train.drop("OutcomeSubtype", axis = 1, inplace = True)

### Get target
target = train["OutcomeType"]

### Drop "OutcomeType" in train, now train and test are in same structure
train.drop("OutcomeType", axis = 1, inplace = True)  # already used as target
train["ID"] = train["AnimalID"]
train.drop("AnimalID", axis = 1, inplace = True)     # unique, independent
target.value_counts()
train.shape
### set index for train and test
train.set_index('ID', inplace=True)
test.set_index('ID', inplace=True)
train.shape

### Combine to data
data = pd.concat([train, test])
### data cleaning
data_cl = data.copy()
data_cl["SexuponOutcome"]["A667395"] = "Neutered Male"

# Color
color_counts = data_cl["Color"].value_counts()
rare_colors = color_counts[color_counts <= 20].index
data_cl.loc[data_cl["Color"].isin(rare_colors), "Color"] = "Other"

# Breed
breed_counts = data_cl["Breed"].value_counts()
rare_breeds = breed_counts[breed_counts <= 20].index
data_cl.loc[data_cl["Breed"].isin(rare_breeds), "Breed"] = "Other"
### Deal with Name later
#data_cl.loc[data_cl["Name"].isnull(), "Name"] = "NoName"
#name_counts = data_cl["Name"].value_counts()
#len(name_counts[name_counts >= 3])
# drop "Name" and "DateTime" temporarily
data_cl.drop("Name", axis=1, inplace=True)
data_cl.drop("DateTime", axis=1, inplace=True)
data_cl.shape

### Encode categorical variables
categorial_cols = ["AnimalType", "SexuponOutcome", "AgeuponOutcome", "Breed", "Color"]

for cc in categorial_cols:
    dummies = pd.get_dummies(data_cl[cc])
    dummies = dummies.add_prefix("{}#".format(cc))
    data_cl.drop(cc, axis=1, inplace=True)
    data_cl = data_cl.join(dummies)
    
data_cl.shape
### Feature Selection

# Seperate to train for validation
X = data_cl.iloc[:26729]
mytest = data_cl.iloc[26729:]

### Use LabelEncoder to encode target labels to numeric
target.value_counts()
le = LabelEncoder()
le.fit(["Adoption","Died","Euthanasia","Return_to_owner","Transfer"])
Y = pd.Series(data = le.transform(target), name = 'target')

'''
# Variance Threshold
threshold = 0.9
vt = VarianceThreshold().fit(X)
feat_var_threshold = data_cl.columns[vt.variances_ > threshold * (1-threshold)]
feat_var_threshold

# RandomForest
rf = RandomForestClassifier().fit(X, Y)
feature_imp = pd.DataFrame(rf.feature_importances_, index = X.columns, columns=["importance"])
feat_imp_20 = feature_imp.sort_values("importance", ascending = False).head(20).index
feat_imp_20


# Recursive Feature Elimination
rfe = RFE(LogisticRegression(), 20).fit(X, Y)
feature_rfe_scoring = pd.DataFrame({
        'feature' : X.columns,
        'score' : rfe.ranking_
    })

feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
feat_rfe_20


# Final feature selection
features = np.hstack([feat_var_threshold, feat_imp_20])
features = np.unique(features)
'''
# Evaluation
seed = 7
processors=1
num_folds=3
num_instances=len(X)
scoring='log_loss'

kfold = KFold(n=num_instances, n_folds=num_folds, random_state=seed)
# Prepare some basic models
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))

#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('K-NN', KNeighborsClassifier(n_neighbors=5)))
#models.append(('SVC', SVC(probability=True)))

# Evaluate each model in turn
results = []
names = []

for name, model in models:
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
    results.append(cv_results)
    names.append(name)
    print("{0}: ({1:.3f}) +/- ({2:.3f})".format(name, cv_results.mean(), cv_results.std()))
'''
categorial_cols = ['AnimalType', 'SexuponOutcome']
for cc in categorial_cols:
    dummies = pd.get_dummies(data_cl[cc])
    dummies = dummies.add_prefix("{}#".format(cc))
    data_cl.drop(cc, axis=1, inplace=True)
    data
'''

'''
### get dummies binary features
dummies = pd.get_dummies(data['AnimalType'])
dummies = dummies.add_prefix("{}#".format('AnimalType'))
dummies2 = pd.get_dummies(data['SexuponOutcome'])
dummies2 = dummies2.add_prefix("{}#".format('SexuponOutcome'))
dummies = pd.concat([dummies, dummies2], axis = 1)

dummies.shape
'''
### Split to train and test from data
#mytrain = dummies.iloc[:26729]
#mytest = dummies.iloc[26729:]

### Use LabelEncoder to encode target labels to numeric
#target.value_counts()
#le = LabelEncoder()
#le.fit(["Adoption","Died","Euthanasia","Return_to_owner","Transfer"])
#mytarget = pd.Series(data = le.transform(target), name = 'target')
### LR not suitable? maybe, i am newbee-.-
#lr = LogisticRegression(solver = "lbfgs", multi_class = "multinomial")
#lr = LogisticRegression()
#lr.fit(X, Y)
#result = lr.predict_proba(mytest)


### GBRT
gbrt = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=5).fit(X, Y)
#result = gbrt.predict_proba(mytest)
### DecisionTreeClassifier first 
#dt = DecisionTreeClassifier()
#dt.fit(mytrain,mytarget)
result = gbrt.predict_proba(mytest)
result
submission = pd.DataFrame()
submission["ID"] = test.index
submission["Adoption"] = result[:, 0]
submission["Died"] = result[:, 1]
submission["Euthanasia"] = result[:, 2]
submission["Return_to_owner"] = result[:, 3]
submission["Transfer"] = result[:, 4]

submission.to_csv("sub.csv", index = False)
#print(check_output(["tail", "sub.csv"]).decode("utf8"))