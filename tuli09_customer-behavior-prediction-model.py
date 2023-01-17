import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split, GridSearchCV

from imblearn.pipeline import Pipeline



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.tree import ExtraTreeClassifier

from xgboost import XGBClassifier



from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from sklearn.inspection import permutation_importance



import pickle

from time import time



import warnings

warnings.filterwarnings("ignore")

sns.set_style('darkgrid')        
train_df = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")

train_df.head()
print("There are total {} samples present in the dataset each with {} features."

      .format(train_df.shape[0], train_df.shape[1]))
train_df.describe()
train_df.select_dtypes(include='object').columns
train_df.select_dtypes(include='object').nunique()
fig, ax = plt.subplots(1,2, figsize=(14,8))

sns.countplot(x='Gender', data=train_df, ax=ax[0])

sns.countplot(x='Gender', hue='Response', data=train_df, ax=ax[1])
print("response:no response ratio for male customers : {}"

.format(len(train_df[(train_df['Gender']=='Male') & (train_df['Response']==1)])/len(train_df[(train_df['Gender']=='Male') & (train_df['Response']==0)])))
print("response:no response ratio for female customers : {}"

.format(len(train_df[(train_df['Gender']=='Female') & (train_df['Response']==1)])/len(train_df[(train_df['Gender']=='Female') & (train_df['Response']==0)])))
plt.figure(figsize=(14,8))

sns.barplot(x='Gender', y='Age', hue='Response', data=train_df)

plt.legend(loc='upper right', bbox_to_anchor=(1.1,1.0))
fig,ax = plt.subplots(1,3,figsize=(14,6))

sns.countplot('Driving_License', data=train_df, ax=ax[0])

sns.countplot('Driving_License', hue='Response', data=train_df, ax=ax[1])

sns.countplot('Driving_License', hue='Previously_Insured', data=train_df, ax=ax[2])
fig,ax = plt.subplots(1,2,figsize=(14,8))

sns.countplot('Previously_Insured', data=train_df, ax=ax[0])

sns.countplot('Previously_Insured', hue='Response', data=train_df, ax=ax[1])
fig, ax = plt.subplots(1,3,figsize=(14,6))

sns.countplot('Vehicle_Damage', data=train_df, ax=ax[0])

sns.countplot('Vehicle_Damage', hue='Response', data=train_df, ax=ax[1])

sns.countplot('Vehicle_Damage', hue='Previously_Insured', data=train_df, ax=ax[2])

plt.tight_layout()
plt.figure(figsize=(12,6))

sns.distplot(train_df['Age'])
fig, ax = plt.subplots(1, 3, figsize=(14,8))

sns.countplot('Vehicle_Age', data=train_df, ax=ax[0])

sns.countplot('Vehicle_Age', hue='Response', data=train_df, ax=ax[1])

sns.countplot('Vehicle_Age', hue='Previously_Insured', data=train_df, ax=ax[2])

plt.tight_layout()
fig, ax = plt.subplots(2,2,figsize=(14,6))

sns.distplot(train_df['Annual_Premium'], ax=ax[0,0])

sns.barplot(x='Response', y='Annual_Premium', data=train_df, ax=ax[0,1])

sns.distplot(train_df[train_df['Response']==0]['Annual_Premium'], ax=ax[1,0])

sns.distplot(train_df[train_df['Response']==1]['Annual_Premium'], ax=ax[1,1])

ax[0,0].set_xlim([0,100000])

ax[1,0].set_xlim([0,100000])

ax[1,1].set_xlim([0,100000])
train_df['Annual_Premium'].mean()
sns.distplot(train_df['Policy_Sales_Channel'])
# top 10 marketting channels used by the company

train_df['Policy_Sales_Channel'].value_counts().head(10).plot(kind='bar', figsize=(14,8))
# top 10 marketting channels for the customers who didn't respond

train_df[train_df['Response']==0]['Policy_Sales_Channel'].value_counts(normalize=True).head(10).plot(kind='bar')
# top 10 marketting channels for the customers who responded

train_df[train_df['Response']==1]['Policy_Sales_Channel'].value_counts(normalize=True).head(10).plot(kind='bar')
train_df.head()
train_df.isnull().sum()
# drop the 'id' column since it won't be used during model traiing

final_train_df = train_df.drop('id', axis=1)
final_train_df[final_train_df.duplicated()]
final_train_df.drop_duplicates(inplace=True)
print("The shape of the dataframe after dropping duplicate rows is : {}".format(final_train_df.shape))
for col in final_train_df.select_dtypes(include='object').columns:

    print(col, ":", final_train_df[col].unique())

    print()
# encoding binary categorical features

final_train_df.replace({'Male':0, 'Female':1, 'No':0, 'Yes':1}, inplace=True)

final_train_df.head()
# create dummy variables for the categorical feature with more than two classes

final_train_df = pd.get_dummies(final_train_df, drop_first=True)

final_train_df.head()
corr_df = final_train_df.corr()

plt.figure(figsize=(12,8))

sns.heatmap(corr_df, annot=True, cmap='coolwarm')
# the feature matrix

X = final_train_df.drop('Response', axis=1)



# the target vector

y = final_train_df['Response']
X.head()
# distribution of classes in the target variable

y.value_counts(normalize=True)
y.value_counts().plot(kind='bar')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("The shape of the training set is {}".format(X_train.shape))

print("The shape of the training set is {}".format(X_test.shape))
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)
def build_pipeline(clf):

    

    """

    Function to build a data pipeline consisting of the following steps :

    

    1. over : oversampling the minority class (response=1) class using SMOTE technique.

    2. under : undersampling the majority class (response=0).

    3. scaler : standardizing the dataset.

    4. clf : the classification algorithm.

    

    Parameter

    ----------

    clf : object of a class

       the classification class object

       

    Returns   

    ---------

    pipeline : object of a class

       the data pipeline object 

    

    """

    

    over = SMOTE(sampling_strategy=0.2)

    under = RandomUnderSampler(sampling_strategy=0.5)

    scaler = StandardScaler()

    

    pipeline = Pipeline([

                        ('over', over),

                        ('under', under),

                        ('scaler', scaler),

                        ('clf', clf)

                       ])

    return pipeline
# list of classifiers to be analyzed

clf_list = [LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier(), 

            AdaBoostClassifier(), XGBClassifier()]



# list of dictionaries of parameters and corresponding values associated with each of the classifiers in the above 

# list that will be used during grid search



param_list = [

             {

              'clf__C':[0.01, 0.05, 0.1, 0.3],

              'clf__class_weight':[None, 'balanced']                

             },

             {

              'clf__n_neighbors':[5, 10, 15, 20],

              'clf__weights':['uniform', 'distance']   

             },

             {

              'clf__n_estimators':[80, 100, 150],

              'clf__criterion':['gini', 'entropy'],

              'clf__class_weight':[None, 'balanced']   

             },

             {

              'clf__n_estimators':[80, 100, 150],

              'clf__learning_rate':[0.001, 0.01, 0.1]   

             },

             {

              'clf__n_estimators':[80, 100, 150],

              'clf__learning_rate':[0.001, 0.01, 0.1]                    

             }

             ]
#roc_score_max = 0.



# loop over classifiers and corresponding dictionary of parameters

#for param_dict, clf in zip(param_list, clf_list):

    

    # build the pipeline

#    pipeline = build_pipeline(clf)

#    print("Now running the model : {}".format(pipeline.steps[3][1]))

    

    # build the GridSearchCV object

#    grid_cv = GridSearchCV(pipeline, param_grid=param_dict, cv=3, scoring='roc_auc', verbose=2)

#    print()

#    initial_time = time()

    

    # fit this object to the training set

#    grid_cv.fit(X_train, y_train)

#    train_time = time() - initial_time

#    print("total time taken for fitting the model : {}".format(train_time))

#    print()

    

    # make predictions on the test set

#    pred = grid_cv.predict_proba(X_test)

    

    # compute the roc_auc score for the test set

#    roc_score = roc_auc_score(y_test,pred[:, 1:])

    

    # store the best roc_score and the corresponding classifier

#    if roc_score > roc_score_max:

#        roc_score_max = roc_score

#        opt_model = pipeline.steps[3][1]

#        opt_param = grid_cv.best_params_

#        opt_val_score, opt_test_score = grid_cv.best_score_, roc_score_max

        

        

#print("the best model is {} for the parameter set {} that produces an roc_auc_score {} when evaluated on the train set and roc_auc_score {} when evaluated on the test set."

#      .format(opt_model, opt_param, opt_val_score, opt_test_score))        
# further optimization of the best model found during grid search

xgb = XGBClassifier()



param_dict = {'clf__n_estimators':list(np.arange(250,400,50)),

              'clf__learning_rate':list(np.round(np.arange(0.1,0.4,0.15),2)),

              'clf__scale_pos_weight':[0.3,0.4,0.5] }





pipeline = build_pipeline(xgb)

grid_cv = GridSearchCV(pipeline, param_grid=param_dict, cv=5, scoring='roc_auc', verbose=2)



initial_time = time()

grid_cv.fit(X_train, y_train)

train_time = time() - initial_time

print("total time taken for fitting the model : {}".format(train_time))

print()

pred = grid_cv.predict_proba(X_test)

roc_score = roc_auc_score(y_test,pred[:, 1:])
print("the best parametet set for the optimized xgb classifier is : {}".format(grid_cv.best_params_))
print("ROC AUC score for the optimized model is {}".format(roc_score))
# save the model

pickle.dump(grid_cv, open("model_xgb.pickle", "wb"))
#grd_boost = GradientBoostingClassifier()



#param_dict = {'clf__n_estimators':list(np.arange(280,350,20)),

#              'clf__learning_rate':list(np.round(np.arange(0.2,0.4,0.05),2))}





#pipeline = build_pipeline(grd_boost)

#grid_cv_grd = GridSearchCV(pipeline, param_grid=param_dict, cv=5, scoring='roc_auc', verbose=2)



#initial_time = time()

#grid_cv_grd.fit(X_train, y_train)

#train_time = time() - initial_time

#print("total time taken for fitting the model : {}".format(train_time))

#print()

#pred_gb = grid_cv_grd.predict_proba(X_test)

#roc_score = roc_auc_score(y_test,pred_gb[:, 1:])
#param_dict = {'clf__n_estimators':list(np.arange(20,100,10)),

#              'clf__learning_rate':list(np.round(np.arange(0.1,0.8,0.1),2)),

#              'clf__base_estimator__min_samples_split':[40, 60, 80, 100],

#              'clf__base_estimator__min_samples_leaf':[40, 60, 80, 100]

#             }



#clf = AdaBoostClassifier(base_estimator=ExtraTreeClassifier())



#pipeline = build_pipeline(clf=clf)

#grid_cv_ada = GridSearchCV(pipeline, param_grid=param_dict, cv=3, scoring='roc_auc', verbose=2)



#initial_time = time()

#grid_cv_ada.fit(X_train, y_train)

#train_time = time() - initial_time

#print("total time taken for fitting the model : {}".format(train_time))



#pred_ada = grid_cv_ada.predict_proba(X_test)

#roc_score = roc_auc_score(y_test,pred_ada[:, 1:])
# fpr and tpr for our optimized model

fpr, tpr, thresholds = roc_curve(y_test,pred[:, 1:])
# plot the ROC curve

plt.figure(figsize=(12,6))

ident = [0.0, 1.0]

plt.plot(fpr, tpr)

plt.plot(ident, ident, linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(['xgbbost', 'noskill model'])

plt.title("ROC Curve")
# permutation importance for feature evaluation on the training set

feat_imp = permutation_importance(grid_cv, X_train, y_train, scoring='roc_auc', n_repeats=10)
pd.DataFrame(feat_imp['importances_mean'], columns=['feature_imp'], index=X_train.columns).sort_values(by='feature_imp').plot(kind='barh', figsize=(14,8))

plt.legend(loc='lower right')

plt.title('Feature importance for XGboost (Training set)')
test_df = pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")

test_df.head()
print("the shape of the test dataset is : {}".format(test_df.shape))
# encoding binary categorical features

test_df.replace({'Male':0, 'Female':1, 'No':0, 'Yes':1}, inplace=True)



# create dummy variables for the categorical feature with more than two classes

final_test_df = pd.get_dummies(test_df, drop_first=True)

final_test_df.head()
# load the trained model

clf_model = pickle.load(open("model_xgb.pickle", "rb")) 
# predict the probability of the minority class (positive response) on the test set

test_pred = clf_model.predict_proba(final_test_df.iloc[:, 1:])[:, 1:]

test_pred
submission_df = pd.read_csv("../input/health-insurance-cross-sell-prediction/sample_submission.csv")

submission_df.head()
submission_df['Response'] = test_pred

submission_df.head(10)
submission_df.to_csv('submission.csv')