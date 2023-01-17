import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape)
print(test.shape)
train.info()
test.info()
train.describe()
train.columns
plt.figure(figsize=(15,15))
sns.heatmap(train.corr())
#函数引用于他处
from scipy.stats import gaussian_kde

def compare_dist(x,y,ax, feature, i=0):
    sns.kdeplot(x[feature], label="train", ax=ax)
    sns.kdeplot(y[feature], label="test", ax=ax)

def numeric_tile(x,y):
    fig, axs = plt.subplots(2, 6, figsize=(24, 12))
    axs = axs.flatten()
    
    for i, (ax, col) in enumerate(zip(axs, y.columns.tolist()[1:])):
        compare_dist(x,y,ax, col, i)
        ax.set_title(col)
    plt.tight_layout()
list1 = ['Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm']
for i in list1:
    for j in list1:
        if i is not j:
            sns.FacetGrid(train, hue="Cover_Type", size=10).map(plt.scatter, i, j).add_legend()
list2 = ['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
for i in list2:
    if i is not 'Elevation':
        sns.FacetGrid(train, hue="Cover_Type", size=10).map(plt.scatter, 
    "Elevation", i).add_legend()
sns.FacetGrid(train, hue="Cover_Type", size=10).map(plt.scatter,"Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology").add_legend()

numeric_tile(train,test)
soil_list = []
for i in range(1, 41):
    soil_list.append('Soil_Type' + str(i))

wilderness_area_list = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']

def wilderness_compress(df):
    
    df[wilderness_area_list] = df[wilderness_area_list].multiply([1, 2, 3, 4], axis=1)
    df['Wilderness_Area'] = df[wilderness_area_list].sum(axis=1)
    df.drop(wilderness_area_list,inplace=True,axis=1)
    return df
def soil_compress(df):
    
    df[soil_list] = df[soil_list].multiply([i for i in range(1, 41)], axis=1)
    df['Soil_Type'] = df[soil_list].sum(axis=1)
    df.drop(soil_list,inplace=True,axis=1)
    return df

def feature_compress(df):
    df = wilderness_compress(df)
    df = soil_compress(df)
    return df
train = feature_compress(train)
train.head()
train_copy = train.copy()
train_label = train_copy["Cover_Type"]
train_copy.drop(["Id","Cover_Type"],inplace=True,axis=1)
train_copy.head()
train_copy = (train_copy - train_copy.min())/(train_copy.max()-train_copy.min())
train_copy.head()
train_copy.describe()
test_Id = test["Id"]
test.drop(["Id"],inplace=True,axis=1)
test = feature_compress(test)
test = (test - test.min())/(test.max()-test.min())
numeric_tile(train_copy,test)
#train['Vertical_Distance_To_Hydrology'] = train['Elevation']-train['Vertical_Distance_To_Hydrology']
#train['Horizontal_Distance_To_Hydrology']=train['Elevation']- train['Horizontal_Distance_To_Hydrology']*0.2
#train['fe_Distance_To_Hydrology'] = np.sqrt(train['Horizontal_Distance_To_Hydrology']**2 + train['Vertical_Distance_To_Hydrology']**2)
#train['fe_Hillshade_Mean'] = (train['Hillshade_9am'] + train['Hillshade_Noon'] + train['Hillshade_3pm'])/3
#train['fe_Hillshade_Mean_Div_E'] = (train['fe_Hillshade_Mean'] / train['Elevation']).clip(upper=255)
#train['fe_Hillshade_Mean_Div_Aspect'] = (train['fe_Hillshade_Mean'] / train['Aspect']).clip(upper=255)
def preprocess(df_):
    #df_.drop('Elevation', axis=1, inplace=True)
    df_['fe_E_Min_02HDtH'] = df_['Elevation']- df_['Horizontal_Distance_To_Hydrology']*0.2
    df_['fe_Distance_To_Hydrology'] = np.sqrt(df_['Horizontal_Distance_To_Hydrology']**2 + 
                                              df_['Vertical_Distance_To_Hydrology']**2)
    
    feats_sub = [('E_Min_VDtH', 'Elevation', 'Vertical_Distance_To_Hydrology'),
                 ('HD_Hydrology_Min_Roadways', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways'),
                 ('HD_Hydrology_Min_Fire', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Fire_Points'),
                 ('Hillshade_9am_Min_Noon', 'Hillshade_9am', 'Hillshade_Noon'),
                 ('Hillshade_Noon_Min_3pm', 'Hillshade_Noon', 'Hillshade_3pm'),
                 ('Hillshade_9am_Min_3pm', 'Hillshade_9am', 'Hillshade_3pm')
                ]
    feats_add = [('E_Add_VDtH', 'Elevation', 'Vertical_Distance_To_Hydrology'),
                 ('HD_Hydrology_Add_Roadways', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways'),
                 ('HD_Hydrology_Add_Fire', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Fire_Points'),
                 ('Hillshade_9am_Add_Noon', 'Hillshade_9am', 'Hillshade_Noon'),
                 ('Hillshade_Noon_Add_3pm', 'Hillshade_Noon', 'Hillshade_3pm'),
                 ('Hillshade_9am_Add_3pm', 'Hillshade_9am', 'Hillshade_3pm')
                ]
    
    for f_new, f1, f2 in feats_sub:
        df_['fe_' + f_new] = df_[f1] - df_[f2]
    for f_new, f1, f2 in feats_add:
        df_['fe_' + f_new] = df_[f1] + df_[f2]
        
    df_['fe_Hillshade_Mean'] = (df_['Hillshade_9am'] + df_['Hillshade_Noon'] + df_['Hillshade_3pm'])/3
    df_['fe_Hillshade_Mean_Div_E'] = (df_['fe_Hillshade_Mean'] / df_['Elevation']).clip(upper=255)
    df_['fe_Hillshade_Mean_Div_Aspect'] = (df_['fe_Hillshade_Mean'] / df_['Aspect']).clip(upper=255)
    
    # A few composite variables
    df_['fe_Hillshade_Ratio1'] = (df_['fe_Hillshade_9am_Min_Noon'] / df_['fe_Hillshade_Noon_Min_3pm']).clip(lower=-5, upper=2)
    df_['fe_Hillshade_Ratio2'] = (df_['fe_Hillshade_9am_Min_3pm']  / df_['Hillshade_Noon']).clip(lower=-2, upper=2)
        
    # The feature is advertised in https://douglas-fraser.com/forest_cover_management.pdf
    df_['fe_Shade9_Mul_VDtH'] = df_['Hillshade_9am'] * df_['Vertical_Distance_To_Hydrology']
    
    # Features inherited from https://www.kaggle.com/leannelong3/r-random-forest
    df_['Elevation_bins50'] = np.floor_divide(df_['Elevation'], 50)
    df_['fe_Horizontal_Distance_To_Roadways_Log'] = np.log1p(df_['Horizontal_Distance_To_Roadways'])

    # this mapping comes from https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info
    climatic_zone = {}
    geologic_zone = {}
    for i in range(1,41):
        if i <= 6:
            climatic_zone[i] = 2
            geologic_zone[i] = 7
        elif i <= 8:
            climatic_zone[i] = 3
            geologic_zone[i] = 5
        elif i == 9:
            climatic_zone[i] = 4
            geologic_zone[i] = 2
        elif i <= 13:
            climatic_zone[i] = 4
            geologic_zone[i] = 7
        elif i <= 15:
            climatic_zone[i] = 5
            geologic_zone[i] = 1
        elif i <= 17:
            climatic_zone[i] = 6
            geologic_zone[i] = 1
        elif i == 18:
            climatic_zone[i] = 6
            geologic_zone[i] = 7
        elif i <= 21:
            climatic_zone[i] = 7
            geologic_zone[i] = 1
        elif i <= 23:
            climatic_zone[i] = 7
            geologic_zone[i] = 2
        elif i <= 34:
            climatic_zone[i] = 7
            geologic_zone[i] = 7
        else:
            climatic_zone[i] = 8
            geologic_zone[i] = 7
            
    df_['Climatic_zone_LE'] = df_['Soil_Type'].map(climatic_zone).astype(np.uint8)
    df_['Geologic_zone_LE'] = df_['Soil_Type'].map(geologic_zone).astype(np.uint8)
    
    for c in df_.columns:
        if c.startswith('fe_'):
            df_[c] = df_[c].astype(np.float32)
    return df_
train = preprocess(train)
train.head()
train_l = train["Cover_Type"]
train.drop(["Id","Cover_Type"],inplace=True,axis=1)
train = (train - train.min())/(train.max()-train.min())
train.describe()
def feature_preprocessing(df):
    df = feature_compress(df)
    df = preprocess(df)
    df = (df - df.min())/(df.max()-df.min())
    return df
train2 = pd.read_csv("../input/train.csv")
test2 = pd.read_csv("../input/test.csv")
train2_label = train2['Cover_Type']
train2.drop(["Id","Cover_Type"],inplace=True,axis=1)
train2 = feature_preprocessing(train2)
test2_id = test2['Id']
test2.drop(["Id"],inplace=True,axis=1)
test2 = feature_preprocessing(test2)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train2,train2_label,test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble

from sklearn.metrics import accuracy_score
preds = pd.DataFrame()
from sklearn.model_selection import GridSearchCV
# best params:{'criterion': 'entropy', 'max_depth': 60, 'max_features': 0.5, 'n_estimators': 300}
# best score:  0.8573318216175358
#clf1 = RandomForestClassifier()
"""
grid_values1={'n_estimators':[300],#[200,300,400,500,600,700,800]
             'max_features':[0.5],#[0.2,0.5,0.8]
             'max_depth':[60],#[50,60,70,80]
             'criterion':["entropy"]#["gini","entropy"]dxs 
}
grid1 = GridSearchCV(clf1,param_grid=grid_values1,cv=5,verbose=1,n_jobs = 4)
grid1.fit(X_train,y_train)
print(grid1.best_params_)
print(grid1.best_score_)

clf1 = RandomForestClassifier(n_estimators=grid1.best_params_["n_estimators"],\
                              max_features=grid1.best_params_["max_features"],\
                              max_depth=grid1.best_params_["max_depth"],\
                              criterion=grid1.best_params_["criterion"])
"""
clf1 = RandomForestClassifier(n_estimators=300,max_features=0.5,max_depth=60,criterion='entropy')
# best params:{'base_estimator__criterion': 'gini', 'base_estimator__max_depth': 80, 'base_estimator__max_features': 0.8, 'base_estimator__n_estimators': 200}
# best score:  0.8738662131519275
"""
clf2 = ensemble.AdaBoostClassifier(ensemble.ExtraTreesClassifier(),n_estimators=250, learning_rate=0.01, algorithm='SAMME')
grid_values2={'base_estimator__n_estimators':[200],#[200,300,400,500,600,700,800]
             'base_estimator__max_features':[0.8],#[0.2,0.5,0.8]
             'base_estimator__max_depth':[80],#[50,60,70,80]
             'base_estimator__criterion':["gini"]#["gini","entropy"]
}
grid2 = GridSearchCV(clf2,param_grid=grid_values2,cv=5,verbose=1,n_jobs = 4)
grid2.fit(X_train,y_train)
print(grid2.best_params_)
print(grid2.best_score_)
clf2 =  ensemble.AdaBoostClassifier(ensemble.ExtraTreesClassifier(n_estimators=grid2.best_params_["base_estimator__n_estimators"],\
                              max_features=grid2.best_params_["base_estimator__max_features"],\
                              max_depth=grid2.best_params_["base_estimator__max_depth"],\
                              criterion=grid2.best_params_["base_estimator__criterion"]),
                                   n_estimators=250, learning_rate=0.01, algorithm='SAMME')
"""
clf2 =  ensemble.AdaBoostClassifier(ensemble.ExtraTreesClassifier(n_estimators=200,max_features=0.8,max_depth=80,criterion='gini'),
                                   n_estimators=250, learning_rate=0.01, algorithm='SAMME')
# best params:{'criterion': 'gini', 'max_depth': 60, 'max_features': 0.8, 'n_estimators': 600}
# best score:  0.8735827664399093
"""
clf3 = ensemble.ExtraTreesClassifier() 
grid_values3={'n_estimators':[600],#[200,300,400,500,600,700,800]
             'max_features':[0.8],#[0.2,0.5,0.8]
             'max_depth':[60],#[50,60,70,80]
             'criterion':["gini"]#["gini","entropy"]
}
grid3 = GridSearchCV(clf3,param_grid=grid_values3,cv=5,verbose=1,n_jobs = 4)
grid3.fit(X_train,y_train)
print(grid3.best_params_)
print(grid3.best_score_)
clf3 =  ensemble.ExtraTreesClassifier(n_estimators=grid3.best_params_["n_estimators"],\
                              max_features=grid3.best_params_["max_features"],\
                              max_depth=grid3.best_params_["max_depth"],\
                              criterion=grid3.best_params_["criterion"])
"""
clf3 =  ensemble.ExtraTreesClassifier(n_estimators=600,max_features=0.8,max_depth=60,criterion='gini')
# best params:{'criterion': 'friedman_mse', 'max_depth': 50, 'max_features': 0.2, 'n_estimators': 500}
# best score:  0.8564814814814815
"""
clf4 = ensemble.AdaBoostClassifier(ensemble.GradientBoostingClassifier(),
                                   n_estimators=250, learning_rate=0.01, algorithm="SAMME")
grid_values4={'base_estimator__n_estimators':[500],#[200,300,400,500,600,700,800]
             'base_estimator__max_features':[0.2],#[0.2,0.5,0.8]
             'base_estimator__max_depth':[50],#[50,60,70,80]
             'base_estimator__criterion':["friedman_mse"]#["friedman_mse","mse","mae"]
}
grid4 = GridSearchCV(clf4,param_grid=grid_values4,cv=5,verbose=1,n_jobs = 4)
grid4.fit(X_train,y_train)
print(grid4.best_params_)
print(grid4.best_score_)
clf4 =  ensemble.AdaBoostClassifier(ensemble.GradientBoostingClassifier(n_estimators=grid4.best_params_["base_estimator__n_estimators"],\
                              max_features=grid4.best_params_["base_estimator__max_features"],\
                              max_depth=grid4.best_params_["base_estimator__max_depth"],\
                              criterion=grid4.best_params_["base_estimator__criterion"]),
                                   n_estimators=250, learning_rate=0.01, algorithm="SAMME")
"""
clf4 =  ensemble.AdaBoostClassifier(ensemble.GradientBoostingClassifier(n_estimators=500,max_features=0.8,
                                                                        max_depth=50,criterion='friedman_mse'),
                                    n_estimators=250, learning_rate=0.01, algorithm="SAMME")
for clf, label in zip([clf1,
                       clf2,
                       clf3, 
                       clf4
                      ], 
                      [
                          'Random Forest',
                          'AdaBoostClassifier_ExtraTreesClassifier',
                          'ExtraTreesClassifier',
                          'AdaBoostClassifier_GradientBoostingClassifier'
                      ]):
    
    clf.fit(X_train,y_train)
    y_hat = clf.predict(X_test)
    preds[label]=y_hat
    test_score = accuracy_score(y_test,y_hat)
    #print("train Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    print("test Accuracy: %0.2f [%s]" % (test_score,  label))
preds
pred_vote = preds.mode(axis=1)
print(accuracy_score(y_test,pred_vote[0]))
test_preds = pd.DataFrame()
for clf,label in zip([clf1,clf2,clf3,clf4],
                     ['Random Forest','AdaBoostClassifier_ExtraTreesClassifier',
                      'ExtraTreesClassifier','AdaBoostClassifier_GradientBoostingClassifier']):
    test_preds[label]=clf.predict(test2)
test_preds
test_pred_vote = test_preds.mode(axis=1)
sub = pd.DataFrame({"Id":test2_id,"Cover_Type": test_pred_vote[0].astype('int').values})
sub.to_csv("sub.csv", index=False)
