# Importing all required libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns



# Stats

from scipy import stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# For ignoring the warnings

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



import gc

gc.enable()
# Fetching the data to pandas dataframes

train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
# Checking training data

train.shape
train.head()
# Checking the datatypes of columns

train.dtypes
train.dtypes.value_counts()
train.isna().sum().sort_values(ascending = False).head()
# Checking if every row has unique id

len(train['ID_code'].unique())
# Target column

train['target'].value_counts()
ax = sns.countplot(x="target",data=train)
import tqdm

train_enc = pd.DataFrame(index = train.index)

for col in tqdm.tqdm_notebook(train.columns):

    train_enc[col] = train[col].factorize()[0]
feature_names = train.columns[2:]

feature_names

# Find skewed numerical features

skew_features = train[feature_names].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))

skewness = pd.DataFrame({'Skew' :high_skew})

skew_features.head(10)
# Dropping ID and target columns

z_score_calc = train.drop(columns=['ID_code', 'target'])

# Calculating z score

z = np.abs(stats.zscore(z_score_calc))

# print(z)

threshold = 3

print(np.where(z > 4))
treated_data = train[(z < 4).all(axis=1)]
print("before treating outliers : {}".format(train.shape))

print("after treating outliers : {}".format(treated_data.shape))
treated_data.columns
# Creating the variables for model fitting

X = treated_data.drop(columns=['ID_code', 'target'])

y = treated_data['target']



# Test variable

test = test.drop(columns=['ID_code'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .3, random_state=0)
## Import the random forest model.

from sklearn.ensemble import RandomForestClassifier

## This line instantiates the model. 

rf = RandomForestClassifier() 

## Fit the model on your training data.

rf.fit(X_train, y_train) 

## And score it on your testing data.

rf.score(X_test, y_test)
prediction_rf = rf.predict(test)
train.columns
submission=pd.DataFrame({"ID_code":sub['ID_code'],

                         "target":prediction_rf})

submission.to_csv('submission_rf.csv',index=False)
feature_importances = pd.DataFrame(rf.feature_importances_,

                                   index = X_train.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances
feature_importances.median()
feature_importances.tail(15)
X_train.drop( ['var_38','var_158','var_73','var_14','var_10','var_84','var_61','var_103','var_185'],axis = 1,inplace = True )

X_test.drop( ['var_38','var_158','var_73','var_14','var_10','var_84','var_61','var_103','var_185'],axis = 1 , inplace = True)

X_train.head()

rf.fit(X_train, y_train) 

## And score it on your testing data.

rf.score(X_test, y_test)
feature_selected_test = test.drop( ['var_38','var_158','var_73','var_14','var_10','var_84','var_61','var_103','var_185'],axis = 1)

feature_selected_test.head()
prediction_rf = rf.predict(feature_selected_test)
submission=pd.DataFrame({"ID_code":sub['ID_code'],

                         "target":prediction_rf})

submission.to_csv('submission_rf2.csv',index=False)