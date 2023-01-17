import pandas as pd

import numpy as np

import matplotlib.pyplot as plt  # data visualization

import seaborn as sns # data visualization

%matplotlib inline



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

all_df = pd.concat([train,test])

full_data = [train, test]

PassengerId =  test['PassengerId']
train.head(3)

train.describe()
train.isnull().sum(axis=0).reset_index()

test.isnull().sum(axis=0).reset_index()
train[train["Embarked"].isnull()]
sns.boxplot(x = "Embarked", y = "Fare", data = all_df[ (all_df["Pclass"] == 1)])

plt.axhline(80)

plt.ylim(0,300)
train.loc[train["Embarked"].isnull(),"Embarked"] = "C"

train[train["Embarked"].isnull()]
test[test["Fare"].isnull()]
sns.boxplot(x = "Embarked", y = "Fare", hue = "Pclass", data = all_df)

plt.ylim(0,300)
test.loc[test["Fare"].isnull(),"Fare"] = all_df[(all_df["Embarked"] == "S") & (all_df["Pclass"] == 3)]["Fare"].median()

test[test["Fare"].isnull()]
sns.distplot(all_df[all_df["Age"].notnull()]["Age"],hist = True)

g = sns.FacetGrid(all_df[all_df["Age"].notnull()], col="Pclass", hue = "Survived")

g = g.map(sns.distplot, "Age", hist = True).add_legend()



g = sns.FacetGrid(all_df[all_df["Age"].notnull()], hue = "Survived")

g = g.map(sns.distplot, "Age", hist = True).add_legend()
import scipy.stats as stats



class1 = all_df[(all_df["Pclass"] == 1) & (all_df["Age"].notnull())]["Age"]

class2 = all_df[(all_df["Pclass"] == 2) & (all_df["Age"].notnull())]["Age"]

class3 = all_df[(all_df["Pclass"] == 3) & (all_df["Age"].notnull())]["Age"]



# Perform the ANOVA

stats.f_oneway(class1,class2,class3)
for i in [1,2,3]:

    age_avg = all_df[(all_df["Pclass"] == i) & (all_df["Age"].notnull())]["Age"].mean()

    age_std = all_df[(all_df["Pclass"] == i) & (all_df["Age"].notnull())]["Age"].std()

    for dataset in full_data:

        age_null_count = dataset[dataset["Pclass"] == i]["Age"].isnull().sum()

        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

        dataset.loc[(dataset["Pclass"] == i) & (dataset["Age"].isnull()), "Age"] = age_null_random_list



dataset['Age'] = dataset['Age'].astype(int)
for dataset in full_data:

    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

    dataset["IsAlone"] = dataset["FamilySize"].apply(lambda x: 1 if x ==1 else 0)

    dataset["Sex"] = dataset["Sex"].apply(lambda x: 0 if x == "female" else 1)

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

    dataset.head(3)
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch']

train = train.drop(drop_elements, axis = 1)

test = test.drop(drop_elements,axis = 1)

train = pd.get_dummies(train,drop_first=True)

test = pd.get_dummies(test,drop_first = True)

train.head(3)
y_train = train.loc[:,"Survived"]

x_train = train.drop(['Survived'], axis=1) # Creates an array of the train data

x_test = test # Creats an array of the test data
from sklearn.cross_validation import KFold

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

import xgboost as xgb

from sklearn.svm import SVC
ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        imp = pd.DataFrame( 

        self.clf.fit(x,y).feature_importances_, 

        columns = [ 'Importance' ] , 

        index = x.columns 

        )

        imp = imp.sort_values( [ 'Importance' ] , ascending = True )

        imp.plot( kind = 'barh' )

        print(imp)
# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }





rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
rf.train(x_train,y_train)

rf_oof_train = rf.predict(x_train)

rf_oof_test = rf.predict(x_test)



gb.train(x_train,y_train)

gb_oof_train = gb.predict(x_train)

gb_oof_test = gb.predict(x_test)



svc.train(x_train,y_train)

svc_oof_train = svc.predict(x_train)

svc_oof_test = svc.predict(x_test)
rf_feature = rf.feature_importances(x_train,y_train)
gb_feature = gb.feature_importances(x_train,y_train)
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'SVC': svc_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel()

    })

base_predictions_train.head(3)



base_predictions_test = pd.DataFrame( {'RandomForest': rf_oof_test.ravel(),

     'SVC': svc_oof_test.ravel(),

      'GradientBoost': gb_oof_test.ravel()

    })

base_predictions_test.head(3)
sns.heatmap(base_predictions_train.corr(),cmap='coolwarm',annot=True)
xgb_params = {

    #'eta': 0.05,

    'max_depth': 8,

    'min_child_weight': 2,

    'subsample': 0.8,

    'gamma': 0.9,

    'colsample_bytree': 0.8,

    'objective': 'binary:logistic',

    'scale_pos_weight':1,

    'nthread': -1,

    'silent': 1

}



dtrain = xgb.DMatrix(base_predictions_train, y_train, feature_names= base_predictions_train.columns.values)

xgb_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= 10)



dtest = xgb.DMatrix(base_predictions_test)

pred_test = xgb_model.predict(dtest)

pred_train = xgb_model.predict(dtrain)



pred_test[0:5]

pred_train[0:5]
pred_test[pred_test >= 0.5] = 1

pred_test[pred_test < 0.5] = 0

pred_train[pred_train >= 0.5] = 1

pred_train[pred_train < 0.5] = 0

pred_test[0:5]

pred_train[0:5]
pred_train_df = pd.DataFrame({'Survived': y_train.values,'predicted': pred_train})

pred_train_df.head(3)
from sklearn.metrics import roc_curve

from sklearn.metrics import auc
# Compute fpr, tpr, thresholds and roc auc

fpr, tpr, thresholds = roc_curve(pred_train_df['Survived'], pred_train_df['predicted'])

roc_auc = auc(fpr, tpr)



# Plot ROC curve

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate or (1 - Specifity)')

plt.ylabel('True Positive Rate or (Sensitivity)')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(pred_train_df['Survived'], pred_train_df['predicted']))

print(classification_report(pred_train_df['Survived'], pred_train_df['predicted']))
# Generate Submission File 

submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': pred_test})

submission["Survived"] = submission["Survived"].astype('int')

submission.head(3)



submission.to_csv("titanic_submission.csv", index=False)
submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': base_predictions_test.iloc[:,1]})

submission["Survived"] = submission["Survived"].astype('int')

submission.head(3)





submission.to_csv("titanic_submission_rf.csv", index=False)
def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,)) ##

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf):#get the train and test index generated by Kfold

        x_tr = x_train.iloc[train_index] #xtrain for a fold

        y_tr = y_train.iloc[train_index] #ytrain for a fold

        x_te = x_train.iloc[test_index]  #xtest for a fold



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te) #includes predicted value for test in fold

        oof_test_skf[i, :] = clf.predict(x_test)  #includes predicted value for true test set



    oof_test[:] = oof_test_skf.mean(axis=0) #includes mean predicted value for true test set based on 5 fold model

    #oof_test[:] = oof_test_skf.mode(axis=0) #why not mode

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }





rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)



rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'SVC': svc_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel()

    })

base_predictions_train.head(3)



base_predictions_test = pd.DataFrame( {'RandomForest': rf_oof_test.ravel(),

     'SVC': svc_oof_test.ravel(),

      'GradientBoost': gb_oof_test.ravel()

    })

base_predictions_test.head(3)
xgb_params = {

    'eta': 0.05,

    'max_depth': 4,

    'min_child_weight': 2,

    'subsample': 0.8,

    'gamma': 0.9,

    'colsample_bytree': 0.8,

    'objective': 'binary:logistic',

    'scale_pos_weight':1,

    'nthread': -1,

    'silent': 1

}



dtrain = xgb.DMatrix(base_predictions_train, y_train, feature_names= base_predictions_train.columns.values)



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=50,

    verbose_eval=5, show_stdv=False)



cv_output[['train-error-mean', 'test-error-mean']].plot()

num_boost_rounds = len(cv_output)

num_boost_rounds



xgb_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= 6)



dtest = xgb.DMatrix(base_predictions_test)

pred_test = xgb_model.predict(dtest)

pred_train = xgb_model.predict(dtrain)



pred_test[0:5]

pred_train[0:5]
pred_test[pred_test >= 0.5] = 1

pred_test[pred_test < 0.5] = 0

pred_train[pred_train >= 0.5] = 1

pred_train[pred_train < 0.5] = 0

pred_test[0:5]

pred_train[0:5]



pred_train_df = pd.DataFrame({'Survived': y_train.values,'predicted': pred_train})

pred_train_df.head(3)
# Generate Submission File 

submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': pred_test})

submission["Survived"] = submission["Survived"].astype('int')

submission.head(3)



submission.to_csv("titanic_submission_cv.csv", index=False)
x_train.columns.values

feature_name = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone',

       'Embarked_Q', 'Embarked_S', 'FareBin1',

       'FareBin2', 'FareBin3',

       'AgeBin1', 'AgeBin2', 'AgeBin3',

       'AgeBin4']
xgb_params = {

    'eta': 0.05,

    'max_depth': 4,

    'min_child_weight': 2,

    'subsample': 0.8,

    'gamma': 0.9,

    'colsample_bytree': 0.8,

    'objective': 'binary:logistic',

    'scale_pos_weight':1,

    'nthread': -1,

    'silent': 1

}



dtrain = xgb.DMatrix(x_train, y_train, feature_names= feature_name)



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=1000,

    verbose_eval=50, show_stdv=False)



cv_output[['train-error-mean', 'test-error-mean']].plot()
xgb_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= 300)



dtest = xgb.DMatrix(x_test,feature_names=feature_name)

pred_test = xgb_model.predict(dtest)

pred_train = xgb_model.predict(dtrain)



pred_test[0:5]

pred_train[0:5]
pred_test[pred_test >= 0.5] = 1

pred_test[pred_test < 0.5] = 0

pred_train[pred_train >= 0.5] = 1

pred_train[pred_train < 0.5] = 0

pred_test[0:5]

pred_train[0:5]



pred_train_df = pd.DataFrame({'Survived': y_train.values,'predicted': pred_train})

pred_train_df.head(3)
# Generate Submission File 

submission_xgb = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': pred_test})

submission_xgb["Survived"] = submission_xgb["Survived"].astype('int')

submission_xgb.head(3)



submission_xgb.to_csv("titanic_submission_xgb.csv", index=False)
base_predictions_test["xgb"] = submission_xgb["Survived"]

base_predictions_test.head()
submission_vote = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': base_predictions_test.mode(axis=1)[0].values})

submission_vote["Survived"] = submission_vote["Survived"].astype('int64')

submission_vote.head(3)



submission_vote.to_csv("titanic_submission_vote.csv", index=False)