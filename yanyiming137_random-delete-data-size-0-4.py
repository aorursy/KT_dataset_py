# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import gc
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
trainlength = round(length * 0.4)
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
#test_id = extracte_feas(test_set)
def standardize(data):
    dataset = data.copy(deep = True)
    mean_std = {}
    for col in dataset.columns:
        mean_std[col] = [np.mean(dataset[col]), np.std(dataset[col])]
        dataset[col] = (dataset[col] - mean_std[col][0]) / mean_std[col][1]
    return dataset, mean_std
    
std_training_set, mean_std = standardize(training_set)
frac_pca_error = []
frac_pca = []
frac_pca_training_error =[]
frac_pca_validation_error = []
frac_tree_training_error = []
frac_tree_validation_error = []
var_frac = [i / 100 for i in range(90,100)] +[1]
# randomly delete percentage of data to see how missing data will influence the result
percentage = [i/10 for i in range(1,10)] + [0.99, 0.999]
a = 1
for frac in percentage:
    print("percentage of dropping training data:", frac)
    print("pca cross-validation")
    frac_training_set, frac_testset, frac_training_label, frac_testlabel = train_test_split(training_set, training_label, train_size = 1-frac, test_size = frac)
    del frac_testset
    del frac_testlabel
    gc.collect()
    frac_training_set = frac_training_set.reset_index(drop = True)
    frac_training_label = frac_training_label.reset_index(drop = True)
    #ridge regression pca cv
    frac_std_training_set, frac_mean_std = standardize(frac_training_set)
    if a == 1:
        pca_cv_error = []
        LR = Ridge(alpha = 10)
        for fra in var_frac:
            pca_features = PCA_features(frac_std_training_set, frac = fra)
            pca_training_set = frac_training_set[pca_features]
            pca_cv_error.append(cv_error(LR, dataset = pca_training_set, datalabel = frac_training_label, cv = 5))
        print("minimum pca cv error:", min(pca_cv_error))
        frac_pca_error.append(min(pca_cv_error))
        best_frac = var_frac[pca_cv_error.index(min(pca_cv_error))]
        print("best fraction of variance:", best_frac)
        frac_pca.append(best_frac)

        pca_features = PCA_features(frac_std_training_set, frac = best_frac)
        pca_training_set = frac_std_training_set[pca_features]
        valset1 = validation_set[pca_features].copy(deep = True)
        for col in valset1.columns:
            valset1[col] = (valset1[col] - frac_mean_std[col][0]) / frac_mean_std[col][1]
        LR.fit(pca_training_set, frac_training_label)
        frac_trainlabel = LR.predict(pca_training_set)
        #frac_trainerror = 1- LR.score(pca_training_set, frac_training_label)
        frac_pca_training_error.append(frac_trainerror)
        frac_vallabel = LR.predict(valset1)
        frac_trainerror = LR_error(frac_trainlabel, frac_training_label)
        frac_valerror = LR_error(frac_vallabel, val_label)
        #frac_valerror = 1 - LR.score(valset1, val_label)
        print("pca validation error", frac_valerror)
        frac_pca_validation_error.append(frac_valerror)
        del valset1
        del pca_training_set
        del frac_trainlabel
        del frac_vallabel
        gc.collect()
    
    # tree regression
    valset2 = validation_set.copy(deep = True)
    for col in frac_std_training_set.columns:
        valset2[col] = (valset2[col] - frac_mean_std[col][0]) / frac_mean_std[col][1]
    rfr = RandomForestRegressor(n_estimators = 100, n_jobs = -1)
    rfr.fit(frac_std_training_set, frac_training_label)
    pred_train = rfr.predict(frac_std_training_set)
    pred_val = rfr.predict(valset2)
    frac_tree_training_error.append(LR_error(pred_train, frac_training_label))
    frac_tree_validation_error.append(LR_error(pred_val, val_label))
    del valset2
    del frac_training_set
    del frac_training_label
    del pred_train
    del pred_val
    del rfr
    gc.collect()
    print()
listname = ["dropping percentage of training data", "pca variance fraction", 
            "pca cross validation error", "pca training error", "pca validation error", "RFR training error", "RFR validation error"]
errorlist = pd.DataFrame([percentage, frac_pca, frac_pca_error, frac_pca_training_error, 
              frac_pca_validation_error, frac_tree_training_error, frac_tree_validation_error],index = listname).T
errorlist.to_csv("errorlist.csv",index=False,sep=',')
errorlist
