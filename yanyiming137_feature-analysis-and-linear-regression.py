# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
#from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
path = "../input/"
files = os.listdir("../input")
# shuffle the dataset and split into training_set and validation_set
test_set = pd.read_csv(path+files[1])
original_dataset = pd.read_csv(path+files[0])
original_features = list(original_dataset.columns)
original_dataset.dropna(inplace = True)   
original_dataset = original_dataset.sample(frac = 1).reset_index(drop = True)

length = original_dataset.shape[0]
trainlength = round(length * 0.7)
#prelength = round(length * 0.2)

#training_set = pd.DataFrame(original_dataset.iloc[prelength:trainlength], columns = original_features).reset_index(drop = True)
training_set = pd.DataFrame(original_dataset.iloc[0:trainlength], columns = original_features).reset_index(drop = True)
validation_set = pd.DataFrame(original_dataset.iloc[trainlength:], columns = original_features).reset_index(drop = True)
#pretraining_set = pd.DataFrame(original_dataset.iloc[0:prelength], columns = original_features).reset_index(drop = True)
del original_dataset
# feature extraction function
def extracte_feas(dataset):
    label = False
    if "winPlacePerc" in dataset.columns:
        dataset_label = dataset["winPlacePerc"]
        dataset.drop("winPlacePerc", axis = 1, inplace = True)
        label = True
    features = list(dataset.columns)
    for fea in features:
        if type(dataset[fea].iloc[0]) == str:
            if fea == "Id":
                string_features = list(dataset[fea])
            dataset.drop(fea, axis = 1, inplace = True)
    if label:
        return dataset_label, string_features
    else:
        return string_features

# pca method to extracte features
def PCA_features(dataset, frac = 1):
    features = list(dataset.columns)
    pca = PCA(n_components = frac)
    pca.fit(dataset)
    #summ = sum(pca.singular_values_)
    #addall = 0
    #for i in range(len(pca.singular_values_)):
        #if round(addall/summ,1) == 0.9:
            #break
        #else:
            #addall += pca.singular_values_[i]
    #extrac_length = i
    correlation = pd.DataFrame(pca.components_,columns = features)
    extrac_columns=[]
    for i in range(correlation.shape[0]):
        #j = correlation.idmax(np.absolute(correlation.iloc[i]))
        j = np.absolute(correlation.iloc[i]).idxmax()
        extrac_columns.append(j)
    #data = pd.DataFrame([dataset[col] for col in extrac_columns]).T
    return extrac_columns
    
# MSE
def LR_error(pred_label, true_label):
    error = 0
    for i in range(len(true_label)):
        error += (pred_label[i] - true_label[i])**2
    error = error / len(true_label)
    return error
    
def cv_error(*estimator, dataset, datalabel, cv = 2):
    length = round(len(datalabel) / cv)
    left = 0
    right = length
    error = 0
    for i in range(cv):
        valerror = 0
        valset = pd.DataFrame(dataset.iloc[left:right], columns = dataset.columns).reset_index(drop = True)
        trainset = dataset.drop([k for k in range(left, right)]).reset_index(drop = True)
        vallabel = datalabel.iloc[left:right].reset_index(drop = True)
        trainlabel = datalabel.drop([k for k in range(left, right)]).reset_index(drop = True)
        for col in trainset.columns:
            mean = np.mean(trainset[col])
            std = np.std(trainset[col])
            trainset[col] = (trainset[col] - mean) / std
            valset[col] = (valset[col] - mean) / std
        if len(estimator) > 1:
            estimator[1].fit(trainset)
            trainset = estimator[1].transform(trainset)
            valset = estimator[1].transform(valset)
        estimator[0].fit(trainset, trainlabel)
        predlabel = estimator[0].predict(valset)
        error += LR_error(predlabel, vallabel)
        left = right
        right += length
        if right > len(datalabel):
            right = len(datalabel)
        del valset, vallabel
        del trainset, trainlabel
            
    return error / cv
            
training_label, training_id = extracte_feas(training_set)
#pre_label, pre_stringfeatures = extracte_feas(pretraining_set)
val_label, val_id = extracte_feas(validation_set)
test_id = extracte_feas(test_set)
def standardize(data):
    dataset = data.copy(deep = True)
    mean_std = {}
    for col in dataset.columns:
        mean_std[col] = [np.mean(dataset[col]), np.std(dataset[col])]
        dataset[col] = (dataset[col] - mean_std[col][0]) / mean_std[col][1]
    return dataset, mean_std
    
std_training_set, mean_std = standardize(training_set)
var_frac = [i / 100 for i in range(90,100)] +[1]
alpha = [0.001, 0.01, 0.1, 1.0, 10, 50, 80, 100, 1000, 5000, 10000]
alpha_error = []
best_alpha = []
ridge_trainingerror = []
ridge_valerror = []
for frac in var_frac:
    alpha_cv_error = []
    print("percentage of variance:", frac)
    pca_features = PCA_features(std_training_set, frac = frac)
    pca_training_set = training_set[pca_features]
    print("ridge cv")
    # ridge cv 
    for alp in alpha:
        LR = Ridge(alpha = alp)
        alpha_cv_error.append(cv_error(LR, dataset = pca_training_set, datalabel = training_label, cv = 5))
    #ridge  
    best_frac_alpha = alpha[alpha_cv_error.index(min(alpha_cv_error))]
    alpha_error.append(min(alpha_cv_error))
    best_alpha.append(best_frac_alpha) 
    del pca_training_set
    pca_training_set = std_training_set[pca_features]
    valset = validation_set[pca_features].copy(deep = True)
    for col in pca_training_set.columns:
        valset[col] = (valset[col] - mean_std[col][0]) / mean_std[col][1]
    print("ridge train")   
    # ridge validation
    LR = Ridge(alpha = best_frac_alpha)
    LR.fit(pca_training_set, training_label)
    pred_train = LR.predict(pca_training_set)
    ridge_trainingerror.append(LR_error(pred_train, training_label))
    pred_val = LR.predict(valset)
    ridge_valerror.append(LR_error(pred_val, val_label))
    del pred_train
    del pred_val
    del pca_features
    del valset
    del pca_training_set
ridgelist = ["percentage of variance", "best alpha", "ridge cross validation error", "ridge training error", "ridge validation error"]
error_ridge = pd.DataFrame([var_frac, best_alpha, alpha_error, ridge_trainingerror, ridge_valerror], index = ridgelist).T
error_ridge.to_csv("error_ridge.csv",index=False,sep=',')
error_ridge
index = ridge_valerror.index(min(ridge_valerror))
var_min = var_frac[index]
alpha_min = best_alpha[index]
LR = Ridge(alpha = alpha_min)
pca_features = PCA_features(std_training_set, frac = var_min)
pca_training_set = std_training_set[pca_features]
testset = test_set[pca_features].copy(deep = True)
for col in testset.columns:
    testset[col] = (testset[col] - mean_std[col][0]) / mean_std[col][1]
LR.fit(pca_training_set, training_label)
pred_test = LR.predict(testset)
ridge_test = pd.DataFrame({"Id": test_id, "winPlacePerc": pred_test})
ridge_test.to_csv("ridge_test.csv",index=False,sep=',')
del testset
del pred_test
del ridge_test
