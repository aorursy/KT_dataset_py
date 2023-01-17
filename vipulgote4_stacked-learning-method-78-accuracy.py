# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# The imports...

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Data handling and analysis

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import KFold, cross_val_score



# Models

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix



from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier



#super learner lib

from mlens.ensemble import SuperLearner



#plotting lib

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install mlens
#let import dataset using pandas



dataset=pd.read_csv('/kaggle/input/performance-prediction/summary.csv')

dataset.head()
dataset.describe()
# copy all names into one variable and put them aside because unique values in tr 

names = dataset["Name"]

dataset.drop(["Name"],inplace=True,axis=1)
dataset.head()




plt.subplots(figsize=(18,14))

sns.heatmap(dataset.corr(),annot=True,linewidths=0.4,linecolor="black",fmt="1.2f",cbar=False)

plt.title("Correlation ",fontsize=50)

plt.xticks(rotation=35)

plt.show()
filterd_col=['GamesPlayed', 'MinutesPlayed', 'PointsPerGame', 'FieldGoalsMade',

       'FieldGoalsAttempt', 'FieldGoalPercent', 'FreeThrowMade', 'FreeThrowAttempt',

       'FreeThrowPercent', 'OffensiveRebounds', 'DefensiveRebounds',

       'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers']
X = dataset[filterd_col]

y = dataset["Target"]


## imported this lib for handling imbalance data

from imblearn.combine import SMOTETomek





smothy = SMOTETomek(random_state = 42)

smothy.fit(X,y)

X_resampled,y_resampled = smothy.fit_resample(X,y)
from sklearn.preprocessing import MinMaxScaler
def get_models():

	models = list()

	models.append(LogisticRegression(solver='liblinear'))

	models.append(DecisionTreeClassifier())

	models.append(SVC(gamma='scale', probability=True))

	models.append(GaussianNB())

	models.append(KNeighborsClassifier())

	models.append(AdaBoostClassifier())

	models.append(BaggingClassifier(n_estimators=10))

	models.append(RandomForestClassifier(n_estimators=10))

	models.append(ExtraTreesClassifier(n_estimators=10))

	models.append(XGBClassifier())

	return models





def get_super_learner(X):

	ensemble = SuperLearner(scorer=accuracy_score, folds=10, shuffle=True, sample_size=len(X)) ## using mlens lib method SuperLearner

	# add base models

	models = get_models()

	ensemble.add(models)

	# add the meta model

	ensemble.add_meta(LogisticRegression(solver='lbfgs'))

	#ensemble.add_meta(XGBClassifier())

	return ensemble




ensemble = get_super_learner(X)

pipeline=Pipeline([

    ('minmax',MinMaxScaler(feature_range=(0,1))),

    ('predict fun',ensemble)

])
X_resampled.dtypes
# converting data-type of gamesplayed col to float type



X_resampled['GamesPlayed1']=X_resampled['GamesPlayed'].astype('float')
X_resampled.pop('GamesPlayed')

# remove this col because it has type int and also we created another same col
# checking data types of each col

X_resampled.dtypes
#pipeline.fit(X_resampled,y_resampled)

## intial tesing of pipeline is working or not
#lets make train test split so that we can evaluate accuracy of classifer



from sklearn.model_selection import train_test_split

train_X,test_X,train_y,test_y=train_test_split(X_resampled,y_resampled,test_size=0.2,random_state = 42)
# shape of training and test data we are going to use

train_X.shape,test_X.shape,train_y.shape,test_y.shape
pipeline.fit(np.array(train_X),np.array(train_y))
pred = pipeline.predict(np.array(test_X))

print('Super Learner: %.3f' % (accuracy_score(test_y, pred) * 100))
plt.subplots(figsize=(14,12))

sns.heatmap(confusion_matrix(test_y, pred),annot=True,fmt="1.0f",cbar=False,annot_kws={"size": 20})

plt.title(f"Super-learner model Accuracy: {accuracy_score(test_y, pred)}",fontsize=40)

plt.xlabel("Target",fontsize=30)

plt.show()
kf = KFold(n_splits=10, shuffle=True, random_state=42)

cv_results = cross_val_score(pipeline, # Pipeline

                                np.array(train_X),np.array(train_y), # Target vector

                                cv=kf, # Cross-validation technique

                                scoring="accuracy", # Loss function

                                n_jobs=-1) # Use all CPU scores

cv_results