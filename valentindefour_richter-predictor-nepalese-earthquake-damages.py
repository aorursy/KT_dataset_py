import numpy as np

import pandas as pd



import pprint



import seaborn as sb

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import xgboost as xgb



from sklearn.metrics import classification_report, f1_score, confusion_matrix



from sklearn import ensemble, tree, linear_model, svm, naive_bayes, neural_network, neighbors



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from IPython.core.interactiveshell import InteractiveShell  

InteractiveShell.ast_node_interactivity = "all"

#allows to, among other functionnalities,print head and info of a df in the same cell 

from IPython.display import display_html 
train = pd.read_csv('/kaggle/input/richters-predictor-modeling-earthquake-damage/train_values.csv')

target = pd.read_csv('/kaggle/input/richters-predictor-modeling-earthquake-damage/train_labels.csv')

test = pd.read_csv('/kaggle/input/richters-predictor-modeling-earthquake-damage/test_values.csv')

sub_format = pd.read_csv('/kaggle/input/richters-predictor-modeling-earthquake-damage/submission_format.csv')
train = pd.merge(train, target, on = 'building_id', how = 'left')

train.set_index('building_id', drop = True, inplace = True)

test.set_index('building_id', drop = True, inplace = True)
train.head()

train.info()
sb.countplot(train['damage_grade'])

print(train['damage_grade'].value_counts())
plt.figure(figsize=(20,5))



plt.subplot(1,3,1)

sb.barplot(train['damage_grade'], train['geo_level_1_id'])



plt.subplot(1,3,2)

sb.barplot(train['damage_grade'], train['geo_level_2_id'])



plt.subplot(1,3,3)

sb.barplot(train['damage_grade'], train['geo_level_3_id'])



plt.show()
plt.figure(figsize=(20,5))



plt.subplot(1,3,1)

sb.distplot(train['age'], kde = False)



plt.subplot(1,3,2)

plt.hist(train['age'], range=(0,200))



plt.subplot(1,3,3)

sb.barplot(train['damage_grade'],train['age'])
plt.figure(figsize = (20,5))



plt.subplot(1,3,1)

sb.barplot(train['damage_grade'], train['height_percentage'])



plt.subplot(1,3,2)

sb.barplot(train['damage_grade'], train['area_percentage'])



plt.subplot(1,3,3)

sb.barplot(train['damage_grade'], train['count_floors_pre_eq'])
superstructure_cols = [x for x in train.columns if 'super' in x]

secondary_use_cols = [x for x in train.columns if 'secondary' in x]



superstructure_corr = train[superstructure_cols+['damage_grade']].corr()

secondary_use_corr = train[secondary_use_cols+['damage_grade']].corr()



plt.figure(figsize=(30,8))



plt.subplot(1,2,1)

sb.heatmap(secondary_use_corr)



plt.subplot(1,2,2)

sb.heatmap(superstructure_corr)
plt.figure(figsize = (20,5))



plt.subplot(1,3,1)

sb.barplot(train['damage_grade'], train['has_superstructure_adobe_mud'])



plt.subplot(1,3,2)

sb.barplot(train['damage_grade'], train['has_superstructure_mud_mortar_stone'])



plt.subplot(1,3,3)

sb.barplot(train['damage_grade'], train['has_superstructure_cement_mortar_brick'])
text_features = []

for column in train.columns:

    if train[column].dtype == 'object':

        text_features.append(column)



for feature in text_features:

    train = train.join(pd.get_dummies(train[feature], prefix = feature))

    test = test.join(pd.get_dummies(test[feature], prefix = feature))

    

    train.drop(feature, axis = 1, inplace = True)

    test.drop(feature, axis = 1, inplace = True)





features = train.drop('damage_grade', axis = 1).columns
train.head()
X_train, X_test, Y_train, Y_test = train_test_split(train[features], train.damage_grade, random_state = 42)
classifiers = [neighbors.KNeighborsClassifier(),

               tree.DecisionTreeClassifier(),

               ensemble.RandomForestClassifier(),

               ensemble.GradientBoostingClassifier(),

               xgb.XGBClassifier()]



def test_models(classifiers):

    

    for model in classifiers:

        

        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)

        

        print(model)

        score = f1_score(Y_test, Y_pred, average='micro')

        print(score)

        print('############')

        

test_models(classifiers)
rf_clf = ensemble.RandomForestClassifier()

xgb_clf = xgb.XGBClassifier()



rf_clf.fit(X_train, Y_train)

y_pred_rf = rf_clf.predict(X_test)



xgb_clf.fit(X_train, Y_train)

y_pred_xgb = xgb_clf.predict(X_test)
df_cm_rf = pd.DataFrame(confusion_matrix(Y_test, y_pred_rf), columns=np.unique(Y_test), index = np.unique(Y_test))

df_cm_rf.index.name = 'Real'

df_cm_rf.columns.name = 'Predicted'



df_cm_xgb = pd.DataFrame(confusion_matrix(Y_test, y_pred_xgb), columns=np.unique(Y_test), index = np.unique(Y_test))

df_cm_xgb.index.name = 'Real'

df_cm_xgb.columns.name = 'Predicted'



plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

sb.heatmap(df_cm_rf, annot=True, fmt='d', annot_kws={"size": 24})



plt.subplot(1,2,2)

sb.heatmap(df_cm_xgb, annot=True, fmt='d', annot_kws={"size": 24})
print("Random Forest")

print(classification_report(Y_test, y_pred_rf))

print('############################################################')

print("XG Boost")

print(classification_report(Y_test, y_pred_xgb))
importance_rf = pd.DataFrame({"Features":features, "Importance_RF":rf_clf.feature_importances_}).sort_values(by='Importance_RF', ascending = False).head(15)

importance_xgb = pd.DataFrame({"Features":features, "Importance_XGB":xgb_clf.feature_importances_}).sort_values(by='Importance_XGB', ascending = False).head(15)



RF_styler = importance_rf.style.set_table_attributes("style='display:inline'").set_caption('Top 15 Random Forest importance')

XGB_styler = importance_xgb.style.set_table_attributes("style='display:inline'").set_caption('Top 15 XGBoost importance')



display_html(RF_styler._repr_html_()+XGB_styler._repr_html_(), raw=True)
train['foundation_type_r'].value_counts()

sb.barplot(train['damage_grade'], train['foundation_type_r'])
num_features = ['geo_level_1_id','geo_level_2_id','geo_level_3_id','age','area_percentage','height_percentage']

i = 1



plt.figure(figsize=(20,10))



for col in num_features:

    plt.subplot(3,3,i)

    ax=sb.boxplot(train[col].dropna())

    plt.xlabel(col)

    i+=1

plt.show()
print('Baseline f1 score :')

print(f1_score(Y_test, y_pred_xgb, average='micro'))

print('Parameters associated :')

xgb_clf.get_params
param_1 = {'max_depth' : [10, 20, 40, 60, 80]}



xgb_gs = GridSearchCV(xgb_clf, param_1, n_jobs=4,verbose=5, scoring='f1_micro', cv=3)



xgb_gs.fit(X_train, Y_train)
def make_submission(test_data, classifier):

    

    classifier.fit(X_train, Y_train)

    

    test_data['damage_grade'] = classifier.predict(test_data[features])



    test_data['damage_grade'].to_csv('submission.csv', index = True)
make_submission(test, xgb_gs)