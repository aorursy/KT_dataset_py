# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

%matplotlib inline

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.info()
train.describe()
sns.violinplot(x = 'Cover_Type', y = 'Elevation', data = train)

#every cover type seems to have different range of elevation, which is a good feature
sns.violinplot(x = 'Cover_Type', y = 'Aspect', data = train)

#this feature is good for type 3 to 7, but 1 and 2 show really similar mean and std for this feature
sns.violinplot(x = 'Cover_Type', y = 'Slope', data = train)

#slope is like aspect across different types 
_, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (15,5)) 
sns.distplot(train['Elevation'], ax=ax1, label = "Skewness: {0:.2f}".format(train['Elevation'].skew())).legend()
sns.distplot(train['Aspect'], ax=ax2 ,label = "Skewness: {0:.2f}".format(train['Aspect'].skew())).legend()
sns.distplot(train['Slope'], ax=ax3, label = "Skewness: {0:.2f}".format(train['Slope'].skew())).legend()
_, axarr = plt.subplots(2 ,2, figsize = (15,10))
sns.violinplot(x = 'Cover_Type', y = 'Horizontal_Distance_To_Hydrology', data = train, ax = axarr[0,0])
sns.violinplot(x = 'Cover_Type', y = 'Vertical_Distance_To_Hydrology', data = train, ax = axarr[0,1])
sns.violinplot(x = 'Cover_Type', y = 'Horizontal_Distance_To_Roadways', data = train, ax = axarr[1,0])
sns.violinplot(x = 'Cover_Type', y = 'Horizontal_Distance_To_Fire_Points', data = train, ax = axarr[1,1])
_, axarr = plt.subplots(2 ,2, figsize = (15,10))
sns.distplot(train['Horizontal_Distance_To_Hydrology'], ax = axarr[0,0], \
             label = "Skewness: {0:.2f}".format(train['Horizontal_Distance_To_Hydrology'].skew())).legend()
sns.distplot(train['Vertical_Distance_To_Hydrology'], ax = axarr[0,1], \
             label = "Skewness: {0:.2f}".format(train['Vertical_Distance_To_Hydrology'].skew())).legend()
sns.distplot(train['Horizontal_Distance_To_Roadways'], ax = axarr[1,0], \
             label = "Skewness: {0:.2f}".format(train['Horizontal_Distance_To_Roadways'].skew())).legend()
sns.distplot(train['Horizontal_Distance_To_Fire_Points'], ax = axarr[1,1], \
             label = "Skewness: {0:.2f}".format(train['Horizontal_Distance_To_Fire_Points'].skew())).legend()
_, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (15,5)) 
sns.barplot(x = 'Cover_Type', y = 'Hillshade_9am', data = train, ax=ax1)
sns.barplot(x = 'Cover_Type', y = 'Hillshade_Noon', data = train, ax=ax2)
sns.barplot(x = 'Cover_Type', y = 'Hillshade_3pm', data = train, ax=ax3)

#hillshade is also good for types 3 to 7, but weak for type 1 and 2
def transform_data(data):
    data['Euclidean_Distance_To_Hydrology'] = np.sqrt(data['Horizontal_Distance_To_Hydrology']**2 + \
                                                     data['Vertical_Distance_To_Hydrology']**2)
    distance_features = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',\
                    'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points',\
                    ]
    #calcualte Euclidean distance with horizontal and vertical distances 
    
    def ratio_big_to_small(a, b): 
        #get ratio of larger of two to smaller of two
        if a < 1:
            a = 1
        if b < 1:
            b = 1
        a, b = np.abs(a), np.abs(b)
        if a >= b:
            return a/b
        else:
            return b/a
        
    for feature1 in distance_features:
        for feature2 in distance_features:
            if feature1 != feature2:
                new_feature1 = feature1 + '*' + feature2
                data[new_feature1] = data[feature1] * data[feature2]
                new_feature2 = feature1 + '+' + feature2
                data[new_feature2] = data[feature1] + data[feature2]
                new_feature3 = feature1 + '-' + feature2
                data[new_feature3] = np.abs(data[feature1] + data[feature2])
                new_feature4 = feature1 + '/' + feature2
                data[new_feature4] = data.apply(lambda row: ratio_big_to_small(row[feature1], row[feature2]), axis=1)
        new_feature5 = feature1 + '^2'
        data[new_feature5] = data[feature1]**2
    #generate new features with existing distance features for the purpose of helping prediction of type 1 and 2
    
    data['Hillshade_Range'] = data[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].apply(np.ptp, axis = 1)
    data['Hillshade_Std'] = data[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].apply(np.std, axis = 1)
    #find relationship between hillshade at different times
    
    return data
train = shuffle(train, random_state = 0)
new_train = transform_data(train)
X_train, y_train = new_train.drop(['Cover_Type'], axis = 1).values, new_train['Cover_Type'].values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
processed_test = scaler.transform(transform_data(test))
def base_grid_search(clf, param, X_train, y_train, metric='accuracy'):
    grid_search = GridSearchCV(clf, param_grid = param, cv = 5, scoring = metric, n_jobs = -1, verbose = 3)
    grid_search.fit(X_train, y_train)
    print (grid_search.best_score_, grid_search.best_params_)

def color_confu(y_true, y_predict):
    #define metric on which optimizations of feature engineerings and algorithms are based upon
    confu_mx = confusion_matrix(y_true, y_predict)
    print('Accuracy for each type: ', confu_mx.diagonal()/confu_mx.sum(axis=1))
    row_sums = confu_mx.sum(axis = 1, keepdims=True)
    norm_conf_mx = confu_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    sns.heatmap(norm_conf_mx, cmap=plt.cm.gray, linewidths=0.1, annot=True)
# a baseline model that got 69% in submisssion as a benchmark to gauge how well the voting classifier performs
extra = ExtraTreesClassifier(max_depth = 85, n_estimators = 65, max_features = 33)
extra.fit(X_train, y_train)
each_type, each_weight = np.unique(extra.predict(processed_test), return_counts=True)
extra_base = ExtraTreesClassifier(max_depth = 85, n_estimators = 65, max_features = 33)
extra_base_predict = cross_val_predict(extra_base, X_train, y_train, cv=5, n_jobs=-1)
color_confu(y_train, extra_base_predict)
# this accuracy is ok, and misclassifications happen at 2<->5, 1<->4, and 1<->2
rf_base = RandomForestClassifier(n_estimators=100, max_depth = 70, max_features = 9)
rf_base_predict = cross_val_predict(rf_base, X_train, y_train, n_jobs = -1, cv=5)
color_confu(rf_base_predict, y_train)
#though its accuracy is a little lower than that of the previous model,
# it is less subject to erros when the actual class is 1 or 2, the diversity the voting classifier needs
#check feature importance with both classifiers
features = new_train.drop(['Cover_Type'], axis=1)
rf_base.fit(X_train, y_train)
extra_base.fit(X_train, y_train)
sorted(zip(rf_base.feature_importances_, list(features)), reverse=True)
sorted(zip(extra_base.feature_importances_, list(features)), reverse=True)
#cut features that have importance values less than feature_threshold in both sets
rf_importance = dict(zip(list(features), rf_base.feature_importances_))
extra_importance = dict(zip(list(features), extra_base.feature_importances_))
trash_features = []
feature_threshold = 0.001
for feature in rf_importance:
    ind = list(new_train).index(feature)
    if rf_importance[feature] < feature_threshold and extra_importance[feature] < feature_threshold:
        trash_features.append(ind)
for feature in extra_importance:
    ind = list(new_train).index(feature)
    if rf_importance[feature] < feature_threshold and extra_importance[feature] < feature_threshold and ind not in trash_features:
        trash_features.append(ind)
X_train_new = np.delete(X_train, trash_features, axis = 1)
new_test = np.delete(processed_test, trash_features, axis = 1)
#take advantage of these two classifiers thorugh combining them with an optimized weights
vote_clf = VotingClassifier(estimators=[('extra', ExtraTreesClassifier(max_depth = 85, n_estimators = 65, max_features = 33,
                                                                     )),
                                        ('rf',RandomForestClassifier(n_estimators=100, max_depth = 70, max_features = 9,
                                                                    ))],
                            voting='soft', weights=[2.3,1])

vote_predict = cross_val_predict(vote_clf, X_train_new, y_train, cv=5, n_jobs=-1)
#this voting classifier achieved 80% on Kaggle, a big improvement over ExtraTree
#this improvement can be attributed to increase in precision of 1,2 classification
#the next step can be extracting more features to differentiate type 1 and type 2
color_confu(y_train, vote_predict)
#code to submit prediction on test data

vote_clf.fit(X_train_new, y_train)
sub = pd.DataFrame({'Id': test['Id'], \
                   'Cover_Type': vote_clf.predict(new_test)})
sub = sub[['Id', 'Cover_Type']]
sub.to_csv('prediction_type.csv', index=False)