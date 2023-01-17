import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



data= pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv', delimiter= ',')

print(np.shape(data))

data.head()
#check for NaNs...

data.isnull().sum().sum()
corr= data.corr().iloc[-1,:].to_numpy().reshape(13,1)

sns.heatmap(corr, yticklabels=data.columns, xticklabels= 0)
#declare the feature values and labels then split data into training/validation sets

from sklearn.model_selection import train_test_split



feats= data.iloc[:,:-1]

labels= data.iloc[:,-1]



x_train, x_devel, y_train, y_devel= train_test_split(feats, labels, test_size= 0.1, random_state= 20)



#train linear model with L1 penalty

from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel



lsvc= LinearSVC(C= 1.0, penalty= 'l1', dual= False).fit(x_train, y_train)

svc_mod= SelectFromModel(lsvc, prefit= True)



#get non-zeroed features 

x_train_svc= svc_mod.transform(x_train) #training set w/non-zeroed features

selected_feats_svc= pd.DataFrame(svc_mod.inverse_transform(x_train_svc), index= x_train.index, columns= x_train.columns)

selected_cols_svc= selected_feats_svc.columns[selected_feats_svc.var() != 0]



#get development set that has only the non-zeroed features

x_devel_svc= x_devel[selected_cols_svc]



#see which features were retained

print('Features retained: ', selected_cols_svc)
from sklearn.feature_selection import SelectKBest, f_classif

kbest_feats= SelectKBest(f_classif, k=5)



#get top 5 best features

x_train_kbest= kbest_feats.fit_transform(x_train, y_train)

selected_feats_kbest= pd.DataFrame(kbest_feats.inverse_transform(x_train_kbest), index= x_train.index, columns= x_train.columns)

selected_cols_kbest= selected_feats_kbest.columns[selected_feats_kbest.var() != 0]



#get development set that has the top 5 features

x_devel_kbest= x_devel[selected_cols_kbest]



#see which features were retained

print('Features retained: ', selected_cols_kbest)
from sklearn.ensemble import RandomForestClassifier



#create and train a random forest

forest= RandomForestClassifier(n_estimators= 1000, random_state= 20)

forest.fit(x_train, y_train)



#get the most important features

forest_feats= SelectFromModel(forest, threshold= 'median')

forest_feats.fit(x_train, y_train)



#get training and development sets that have only the most important features

x_train_forest= forest_feats.transform(x_train)

x_devel_forest= forest_feats.transform(x_devel)



#see which features were retained

for i in forest_feats.get_support(indices= True):

    print(x_train.columns[i])
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import recall_score, precision_score



def eval_svm(train_feats, test_feats, train_labs, test_labs):

    """

    INPUT: train_feats and test_feats are either 2D numpy arrays or pd dataframes with the feature values for the train/test sets 

    train_labs, test_labs are either 1D numpy arrays or pd series with the corresponding labels to the train/test features

    

    OUTPUT: classification_results is a string of results, incl. precision, recall, and f1-score for each class

    """

    #scale features before using in SVM

    scaler= StandardScaler()

    train_feats_scale= scaler.fit_transform(train_feats)

    test_feats_scale= scaler.transform(test_feats)

    

    svm= SVC()

    svm.fit(train_feats_scale, train_labs)

    

    predicts= svm.predict(test_feats_scale)

    

    precision= precision_score(test_labs, predicts, average= None, zero_division= 0)

    recall= recall_score(test_labs, predicts, average= None, zero_division= 0)

    

    return precision, recall



#get performance of model that uses all features

prec_allfeats, rec_allfeats= eval_svm(x_train, x_devel, y_train, y_devel)



#get performance of model that uses features from L1 regularization

prec_svc, rec_svc= eval_svm(x_train_svc, x_devel_svc, y_train, y_devel)



#get performance of model that uses features from SelectKBest

prec_kbest, rec_kbest= eval_svm(x_train_kbest, x_devel_kbest, y_train, y_devel)



#get performance of model that uses features from random forest

prec_forest, rec_forest= eval_svm(x_train_forest, x_devel_forest, y_train, y_devel)



print('SVM precision and recall, all features: ', prec_allfeats, rec_allfeats, '\n'+'SVM precision and recall, L1 regularization features: ', prec_svc, rec_svc)

print('SVM precision and recall, top 5 features: ', prec_kbest, rec_kbest, '\n'+'SVM precision and recall, random forest features: ', prec_forest, rec_forest, '\n')   

print('Average recalls: ', np.mean(rec_allfeats), np.mean(rec_svc), np.mean(rec_kbest), np.mean(rec_forest))