import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# apply ignore

import warnings

warnings.filterwarnings('ignore')
#load train data

train_data = pd.read_csv('../input/learn-together/train.csv')

train_data.head()
import eli5

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from eli5.sklearn import PermutationImportance



# define api



def get_model():

    return RandomForestClassifier(n_estimators=1000,

                                   criterion='gini',

                                   max_features='auto',

                                   random_state=42,

                                   max_depth=50,

                                   bootstrap=False)



def get_score(model, X, y):

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    model.fit(train_X, train_y)

    preds = model.predict(val_X)

    # validate model

    print('val size {0}, mse {1}'.format(len(val_y), mean_absolute_error(val_y, preds)))



def get_importance(model, X, y):

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    model.fit(train_X, train_y)

    return PermutationImportance(model, random_state=1).fit(val_X, val_y)
# 'Id' is a false feature, although adding it makes smaller mse error

forest_features = [cname for cname in train_data.columns if cname not in ['Id','Cover_Type']]

X = train_data[forest_features]

y = train_data.Cover_Type



my_model = get_model()

get_score(my_model, X, y)
def get_feature_importances(model, feature_list, treshold):

    # Get numerical feature importances

    importances = list(model.feature_importances_)

    # List of tuples with variable and importance

    feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(feature_list, importances)]

    # Sort the feature importances by most important first

    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    # Print out the feature and importances 

    return [(key, value) for (key, value) in feature_importances if value > treshold]



importances = get_feature_importances(my_model, X.columns.tolist(), 0.005)

print(*['Variable: {:40} Importance: {}'.format(k,v) for k,v in importances], sep = "\n")

# show feature importance

perm = get_importance(my_model, X, y)

eli5.show_weights(perm, feature_names = X.columns.tolist()) 
# select manually features with importance higher than 0.005

select_features = ['Elevation', 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Hydrology',

'Wilderness_Area4','Soil_Type3','Wilderness_Area1','Vertical_Distance_To_Hydrology','Soil_Type10',

'Hillshade_9am','Hillshade_Noon','Soil_Type39','Aspect','Soil_Type38','Soil_Type4','Soil_Type32','Wilderness_Area3']
X = train_data[select_features]

y = train_data.Cover_Type



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = get_model()

model.fit(train_X, train_y)
import shap 



# Create object that can calculate shap values

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(val_X)
shap.summary_plot(shap_values, val_X)
shap.summary_plot(shap_values[0], val_X, title='Hi')        
shap.summary_plot(shap_values[1], val_X)
shap.summary_plot(shap_values[2], val_X)
shap.summary_plot(shap_values[3], val_X)
shap.summary_plot(shap_values[4], val_X)
shap.summary_plot(shap_values[5], val_X)
shap.summary_plot(shap_values[6], val_X)
preds = model.predict(val_X)

# validate model

print('val size {0}, mse {1}'.format(len(val_y), mean_absolute_error(val_y, preds)))

test_model = get_model()



# use select_features

X = train_data[select_features]

y = train_data.Cover_Type

test_model.fit(X, y) # retrain on full dataset



# read test data

test_data = pd.read_csv('../input/learn-together/test.csv')

test_X = test_data[select_features]



# make predictions used to submit. 

test_preds = test_model.predict(test_X)



# The lines below shows how to save predictions in competition format



output = pd.DataFrame({'Id': test_data.Id,

                       'Cover_Type': test_preds})

output.to_csv('submission.csv', index=False)
output.head()