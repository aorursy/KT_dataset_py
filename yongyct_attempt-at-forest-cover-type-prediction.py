%%time

from datetime import datetime as dt



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import GridSearchCV, cross_val_score



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier
%%time

### Read Data ###

df = pd.read_csv('../input/forest-cover-type-prediction/train.csv')

#### Check Null Data ###

if df[df.isnull().any(axis=1) == True].shape[0] != 0:

    print('Warning, null data present')



### Transform / Wrangle Data ###

X_train = df.iloc[:, :-1]

Y_train = df.iloc[:, -1]



X_test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')

X_test_ids = X_test.iloc[:, 0]
class FeatureTransformer(TransformerMixin):

    '''

    Helper class for transforming input dataframes into desired input features. 

    Implements the feature engineering logic.

    '''

    def __init__(self):

        pass

    

    def fit(self, X):

        ignore_cols = ['Id']

        for col in X.columns:

            if X[col].std() == 0:

                print('Columns to drop: {}, std={}'.format(col, X[col].std()))

                ignore_cols.append(col)

        self.ignore_cols = ignore_cols

        return self

    

    def transform(self, X):

        X = X.copy()

        self.__clean_columns(X)

        return X



    def __clean_columns(self, X):

        drop_cols = self.ignore_cols

        for col in drop_cols:

            if col not in X.columns:

                drop_cols.remove(col)

        X.drop(labels=self.ignore_cols, axis=1, inplace=True)
def predict_results(estimator, X_test, X_test_ids):

    '''

    Helper function for predicting and saving test results

    '''

    Y_Pred = pd.DataFrame(estimator.predict(X_test), columns=['Cover_Type'])

    results = pd.concat([X_test_ids, Y_Pred], axis=1)

    results.to_csv('../input/forest-cover-type-prediction/submission.csv', index=False)
def get_feature_importances(estimator, X):

    return pd.DataFrame(

        np.array([X.columns, estimator.feature_importances_]).T, 

        columns=['Features', 'Importance']

    ).sort_values(by='Importance', ascending=False)
feature_transformer = FeatureTransformer()

X_train = feature_transformer.fit_transform(X_train)

X_test = feature_transformer.transform(X_test)
X_train.head()
# %%time

# lrc = LogisticRegression()

# param_grid = [

#     {

#         'n_jobs': [2],

#         'solver': ['lbfgs', 'saga'],

#         'tol': [1e-4, 1e-5],

#         'C': [0.5, 1, 5],

#         'multi_class': ['auto']

#     }

# ]



# gscv = GridSearchCV(estimator=lrc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=2)

# gscv.fit(X_train, Y_train)

# print('Best Params: ', gscv.best_params_)

# print('Best Score: ', gscv.best_score_)



# lrc = gscv.best_estimator_
# %%time

# svc = SVC()

# param_grid = [

#     {

#         'kernel': ['linear', 'rbf'],

#         'tol': [1e-4, 0.001],

#         'C': [0.5, 1, 5],

#         'gamma': ['scale', 'auto']

#     }

# ]



# gscv = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=2)

# gscv.fit(X_train, Y_train)

# print('Best Params: ', gscv.best_params_)

# print('Best Score: ', gscv.best_score_)



# svc = gscv.best_estimator_
# %%time

# rfc = RandomForestClassifier()

# param_grid = [

#     {

#         'n_jobs': [2],

#         'criterion': ['gini', 'entropy'], 

#         'n_estimators': [200, 500, 700], 

#         'max_depth': [3, 15, 30, None],

#         'max_features': [0.3, 0.6, 'auto']

#     }

# ]



# gscv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=2)

# gscv.fit(X_train, Y_train)

# print('Best Params: ', gscv.best_params_)

# print('Best Score: ', gscv.best_score_)



# rfc = gscv.best_estimator_
# %%time

# etc = ExtraTreesClassifier()

# param_grid = [

#     {

#         'n_jobs': [2],

#         'criterion': ['gini', 'entropy'], 

#         'n_estimators': [200, 500, 700], 

#         'max_depth': [3, 15, 30, None],

#         'max_features': [0.3, 0.6, 'auto']

#     }

# ]



# gscv = GridSearchCV(estimator=etc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=2)

# gscv.fit(X_train, Y_train)

# print('Best Params: ', gscv.best_params_)

# print('Best Score: ', gscv.best_score_)



# etc = gscv.best_estimator_
# %%time

# lgbmc = LGBMClassifier()

# param_grid = [

#     {

#         'n_jobs': [4],

#         'max_depth': [2, 3, -1], 

#         'n_estimators': [150, 200, 250], 

#         'num_leaves': [31, 45, 63, 67],

#         'learning_rate': [0.15, 0.2, 0.25],

#         'reg_lambda': [0, 1.5]

#     }

# ]



# gscv = GridSearchCV(estimator=lgbmc, param_grid=param_grid, n_jobs=4, scoring='accuracy', cv=5)

# gscv.fit(X_train, Y_train)

# print('Best Params: ', gscv.best_params_)

# print('Best Score: ', gscv.best_score_)



# lgbmc = gscv.best_estimator_
# %%time

# xgbc = XGBClassifier()

# param_grid = [

#     {

#         'n_jobs': [4],

#         'max_depth': [2, 3, 10, len(X_train.columns)],

#         'n_estimators': [50, 100, 200], 

#         'reg_lambda': [0, 1.6]

#     }

# ]



# gscv = GridSearchCV(estimator=xgbc, param_grid=param_grid, n_jobs=4, scoring='accuracy', cv=5)

# gscv.fit(X_train, Y_train)

# print('Best Params: ', gscv.best_params_)

# print('Best Score: ', gscv.best_score_)



# xgbc = gscv.best_estimator_
%%time

lrc = LogisticRegression(solver='lbfgs', multi_class='auto')

svc = SVC(gamma='scale')

rfc = RandomForestClassifier(criterion='entropy', max_features=0.6, n_estimators=500, n_jobs=6)

etc = ExtraTreesClassifier(criterion='entropy', max_features=0.6, n_estimators=500, n_jobs=6)

lgbmc = LGBMClassifier(learning_rate=0.2, n_estimators=200, num_leaves=63, n_jobs=6)

xgbc = XGBClassifier(max_depth=2, n_estimators=50, reg_lambda=1.6, tree_method='hist', n_jobs=6)



print('LogisticRegression Accuracy: ', cross_val_score(estimator=lrc, X=X_train, y=Y_train, scoring='accuracy', cv=3))

print('SVC Accuracy: ', cross_val_score(estimator=svc, X=X_train, y=Y_train, scoring='accuracy', cv=3))

print('RandomForestClassifier Accuracy: ', cross_val_score(estimator=rfc, X=X_train, y=Y_train, scoring='accuracy', cv=3))

print('ExtraTreesClassifier Accuracy: ', cross_val_score(estimator=etc, X=X_train, y=Y_train, scoring='accuracy', cv=3))

print('LGBMClassifier Accuracy: ', cross_val_score(estimator=lgbmc, X=X_train, y=Y_train, scoring='accuracy', cv=3))

print('XGBClassifier Accuracy: ', cross_val_score(estimator=xgbc, X=X_train, y=Y_train, scoring='accuracy', cv=3))
%%time

etc = ExtraTreesClassifier(criterion='entropy', max_features=0.6, n_estimators=500, n_jobs=6)

# Fitting best estimator

etc.fit(X_train, Y_train)

# Predicting and getting output prediction file

predict_results(estimator=etc, X_test=X_test, X_test_ids=X_test_ids)
get_feature_importances(etc, X_train).head(10)
print(X_train.columns)

print(X_train.columns.shape)
X_soil = X_train.loc[:, ['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',

       'Soil_Type6', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',

       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']]

X_wild_area = X_train.loc[:, ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']]

X_incline = X_train.loc[:, ['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']]

X_spatial = X_train.loc[:, ['Elevation', 'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 

                      'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']]



X_soil_test = X_test.loc[:, ['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',

       'Soil_Type6', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',

       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']]

X_wild_area_test = X_test.loc[:, ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']]

X_incline_test = X_test.loc[:, ['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']]

X_spatial_test = X_test.loc[:, ['Elevation', 'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 

                      'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']]
# %%time

# ### Soil Type RF Classifier ###

# rfc_soil = RandomForestClassifier()

# #### Perform GridSearchCV to optimize params ####

# rfc_soil_param_grid = [

#     {

#         'n_jobs': [6],

#         'n_estimators': [10, 100, 150],

#         'max_depth': [3, 5, None],

#         'criterion': ['gini', 'entropy']

#     }

# ]



# rfc_gscv = GridSearchCV(

#     estimator=rfc_soil, 

#     param_grid=rfc_soil_param_grid, 

#     scoring='neg_log_loss', 

#     cv=5, n_jobs=6

# )

# rfc_gscv.fit(X_soil, Y)



# print('RFC Soil Best Params: ', rfc_gscv.best_params_)

# print('RFC Soil Best Score: ', rfc_gscv.best_score_)



# #### Get best estimator and predict proba ####

# rfc_soil = rfc_gscv.best_estimator_

# Y_proba_soil_test = rfc_soil.predict_proba(X_soil_test)
# %%time

# ### Wilderness Area RF Classifier ###

# rfc_wild_area = RandomForestClassifier()

# #### Perform GridSearchCV to optimize params ####

# rfc_wild_area_param_grid = [

#     {

#         'n_jobs': [6],

#         'n_estimators': [75, 100, 125],

#         'max_depth': [2, None],

#         'criterion': ['gini', 'entropy']

#     }

# ]



# rfc_wild_area_gscv = GridSearchCV(

#     estimator=rfc_wild_area, 

#     param_grid=rfc_wild_area_param_grid, 

#     scoring='neg_log_loss', 

#     cv=5, n_jobs=6

# )

# rfc_wild_area_gscv.fit(X_wild_area, Y)



# print('RFC Wilderness Best Params: ', rfc_wild_area_gscv.best_params_)

# print('RFC Wilderness Best Score: ', rfc_wild_area_gscv.best_score_)



# #### Get best estimator and predict proba ####

# rfc_wild_area = rfc_wild_area_gscv.best_estimator_

# Y_proba_wild_area_test = rfc_wild_area.predict_proba(X_wild_area_test)
# %%time

# ### Inclination RF Classifier ###

# rfc_incline = RandomForestClassifier()

# #### Perform GridSearchCV to optimize params ####

# rfc_incline_param_grid = [

#     {

#         'n_jobs': [6],

#         'n_estimators': [10, 100, 150, 200],

#         'max_depth': [3, 5, None],

#         'criterion': ['gini', 'entropy']

#     }

# ]



# rfc_incline_gscv = GridSearchCV(

#     estimator=rfc_incline, 

#     param_grid=rfc_incline_param_grid, 

#     scoring='neg_log_loss', 

#     cv=5, n_jobs=6

# )

# rfc_incline_gscv.fit(X_incline, Y)



# print('RFC Inclination Best Params: ', rfc_incline_gscv.best_params_)

# print('RFC Inclination Best Score: ', rfc_incline_gscv.best_score_)



# #### Get best estimator and predict proba ####

# rfc_incline = rfc_incline_gscv.best_estimator_

# Y_proba_incline_test = rfc_incline.predict_proba(X_incline_test)
# %%time

# ### Inclination RF Classifier ###

# lgbmc_spatial = LGBMClassifier()

# #### Perform GridSearchCV to optimize params ####

# lgbmc_spatial_param_grid = [

#     {

#         'n_jobs': [6],

#         'n_estimators': [200, 250, 275],

#         'learning_rate': [0.125, 0.15, 0.175, 0.2],

#         'num_leaves': [65, 67, 70]

#     }

# ]



# lgbmc_spatial_gscv = GridSearchCV(

#     estimator=lgbmc_spatial, 

#     param_grid=lgbmc_spatial_param_grid, 

#     scoring='neg_log_loss', 

#     cv=5, n_jobs=6

# )

# lgbmc_spatial_gscv.fit(X_spatial, Y)



# print('LGBMC Spatial Best Params: ', lgbmc_spatial_gscv.best_params_)

# print('LGBMC Spatial Best Score: ', lgbmc_spatial_gscv.best_score_)



# #### Get best estimator and predict proba ####

# lgbmc_spatial = lgbmc_spatial_gscv.best_estimator_

# Y_proba_spatial_test = lgbmc_spatial.predict_proba(X_spatial_test)
class SegmentClassifier(BaseEstimator, TransformerMixin):

    

    def __init__(self, classifier, columns):

        self.classifier = classifier

        self.columns = columns

    

    def fit(self, X, y):

        X = X.loc[:, self.columns]

        self.classifier.fit(X, y)

        return self

    

    def predict(self, X):

        X = X.loc[:, self.columns]

        return self.classifier.predict(X)

    

    def predict_proba(self, X):

        X = X.loc[:, self.columns]

        return self.classifier.predict_proba(X)
%%time

soil_classifier = SegmentClassifier(

    classifier=RandomForestClassifier(criterion='entropy', n_estimators=100, n_jobs=6),

    columns=['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',

       'Soil_Type6', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',

       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

)



wild_area_classifier = SegmentClassifier(

    classifier=RandomForestClassifier(criterion='gini', n_estimators=100, n_jobs=6),

    columns=['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']

)



incline_classifier = SegmentClassifier(

    classifier=RandomForestClassifier(criterion='entropy', n_estimators=150, max_depth=5, n_jobs=6),

    columns=['Aspect', 'Slope', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']

)



spatial_classifier = SegmentClassifier(

    classifier=LGBMClassifier(learning_rate=0.125, n_estimators=200, num_leaves=65, n_jobs=6),

    columns=['Elevation', 'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 

                      'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']

)
# ensemble_classifier = VotingClassifier(

#     estimators=[

#         ('soil_classifier', soil_classifier),

#         ('wild_area_classifier', wild_area_classifier),

#         ('incline_classifier', incline_classifier),

#         ('spatial_classifier', spatial_classifier)

#     ]

# )
# %%time

# param_grid = [

#     {

#         'voting': ['soft', 'hard'],

#         'weights': [[1,1,2,16], [1,2,3,4], [1,1,4,10]]

#     }

# ]

# gscv = GridSearchCV(ensemble_classifier, param_grid=param_grid, n_jobs=4, cv=5)

# gscv.fit(X_train, Y_train)

# print('Best Params: ', gscv.best_params_)

# print('Best Score: ', gscv.best_score_)
# predict_results(estimator=gscv.best_estimator_, X_test=X_test, X_test_ids=X_test_ids)
### Method 1 ###

ensemble_classifier = VotingClassifier(

    estimators=[

        ('soil_classifier', soil_classifier),

        ('wild_area_classifier', wild_area_classifier),

        ('incline_classifier', incline_classifier),

        ('spatial_classifier', spatial_classifier),

        ('original_classifier', etc)

    ]

)
%%time

param_grid = [

    {

        'voting': ['soft'],

        'weights': [[1,1,2,3,5], [1,2,3,4,5], [0,0,0,0,1], [0,0,0,2,5]]

#         'weights': [[0,0,0,3,5], [0,0,0,2,5]]

    }

]

gscv = GridSearchCV(ensemble_classifier, param_grid=param_grid, n_jobs=6, cv=5)

gscv.fit(X_train, Y_train)

print('Best Params: ', gscv.best_params_)

print('Best Score: ', gscv.best_score_)
predict_results(estimator=gscv.best_estimator_, X_test=X_test, X_test_ids=X_test_ids)
class FeatureTransformer(TransformerMixin):

    '''

    Implementing __enhance_columns method to add more sophisticated features.

    '''

    def __init__(self):

        pass

    

    def fit(self, X):

        ignore_cols = ['Id']

        for col in X.columns:

            if X[col].std() == 0:

                print('Columns to drop: {}, std={}'.format(col, X[col].std()))

                ignore_cols.append(col)

        self.ignore_cols = ignore_cols

        return self

    

    def transform(self, X):

        X = X.copy()

        self.__clean_columns(X)

        self.__enhance_columns(X)

        return X



    def __clean_columns(self, X):

        drop_cols = self.ignore_cols

        for col in drop_cols:

            if col not in X.columns:

                drop_cols.remove(col)

        X.drop(labels=self.ignore_cols, axis=1, inplace=True)

        

    def __enhance_columns(self, X):

        X.loc[:, 'Distance_To_Hydrology'] = (X.loc[:, 'Horizontal_Distance_To_Hydrology'] ** 2 

            + X.loc[:, 'Vertical_Distance_To_Hydrology'] ** 2) ** 0.5

        X.loc[:, 'Distance_To_Amenities_Avg'] = X.loc[:, [

            'Horizontal_Distance_To_Hydrology', 

            'Horizontal_Distance_To_Roadways', 

            'Horizontal_Distance_To_Fire_Points'

        ]].mean(axis=1)

        X.loc[:, 'Elevation_Minus_Disthy'] = X.loc[:, 'Elevation'] - X.loc[:, 'Vertical_Distance_To_Hydrology']

        X.loc[:, 'Elevation_Plus_Disthy'] = X.loc[:, 'Elevation'] + X.loc[:, 'Vertical_Distance_To_Hydrology']

        X.loc[:, 'Disthx_Minus_Distfx'] = X.loc[:, 'Horizontal_Distance_To_Hydrology'] - X.loc[:, 'Horizontal_Distance_To_Fire_Points']

        X.loc[:, 'Disthx_Plus_Distfx'] = X.loc[:, 'Horizontal_Distance_To_Hydrology'] + X.loc[:, 'Horizontal_Distance_To_Fire_Points']

        X.loc[:, 'Disthx_Minus_Distrx'] = X.loc[:, 'Horizontal_Distance_To_Hydrology'] - X.loc[:, 'Horizontal_Distance_To_Roadways']

        X.loc[:, 'Disthx_Plus_Distrx'] = X.loc[:, 'Horizontal_Distance_To_Hydrology'] + X.loc[:, 'Horizontal_Distance_To_Roadways']

        X.loc[:, 'Distfx_Minus_Distrx'] = X.loc[:, 'Horizontal_Distance_To_Fire_Points'] - X.loc[:, 'Horizontal_Distance_To_Roadways']

        X.loc[:, 'Distfx_Minus_Distrx'] = X.loc[:, 'Horizontal_Distance_To_Fire_Points'] - X.loc[:, 'Horizontal_Distance_To_Roadways']
%%time

feature_transformer_new = FeatureTransformer()

X_train = feature_transformer_new.fit_transform(X_train)

X_test = feature_transformer_new.transform(X_test)



etc = ExtraTreesClassifier(criterion='entropy', max_features=0.6, n_estimators=500, n_jobs=6)

# Fitting best estimator

etc.fit(X_train, Y_train)

# Predicting and getting output prediction file

predict_results(estimator=etc, X_test=X_test, X_test_ids=X_test_ids)
get_feature_importances(etc, X_train).head(10)