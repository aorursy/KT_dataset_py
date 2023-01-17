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
trainlength = round(length * 0.2)
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
#tree_number = [i*10 for i in range(5, 11)]
tree_number = [80, 90, 100, 110, 120]
del training_set
gc.collect()
# tree number cv
tree_trainingerror = []
tree_valerror = []
#valset = validation_set.copy(deep = True)
#testset = test_set.copy(deep = True)
for col in std_training_set.columns:
    #valset[col] = (valset[col] - mean_std[col][0]) / mean_std[col][1]
    #testset[col] = (testset[col] - mean_std[col][0]) / mean_std[col][1]
    validation_set[col] = (validation_set[col] - mean_std[col][0]) / mean_std[col][1]
    test_set[col] = (test_set[col] - mean_std[col][0]) / mean_std[col][1]
for num in tree_number:
    print(num)
    rfr = RandomForestRegressor(n_estimators = num, n_jobs = -1)
    rfr.fit(std_training_set, training_label)
    #pred_train = rfr.predict(std_training_set)
    #pred_val = rfr.predict(valset)
    pred_test = rfr.predict(test_set)
    #pred_val = rfr.predict(validation_set)
    tree_trainingerror.append(1-rfr.score(std_training_set, training_label))
    tree_valerror.append(1-rfr.score(validation_set, val_label))
    tree_test = pd.DataFrame({"Id": test_id, "winPlacePerc": pred_test})
    tree_test.to_csv("tree_test"+str(num)+".csv",index=False,sep=',')
    #del pred_train
    #del pred_val
    del rfr
    gc.collect()
#del valset
#del testset
#gc.collect()
    
best_tree = tree_number[tree_valerror.index(min(tree_valerror))]
#treename = ["best tree number", "training error", "validation error"]
treename = ["tree number","training error", "validation error"]
error_tree = pd.DataFrame([tree_number, tree_trainingerror, tree_valerror], index = treename).T
error_tree.to_csv("error_tree.csv",index=False,sep=',')
#tree_test = pd.DataFrame({"Id": test_id, "winPlacePerc": pred_test})
#tree_test.to_csv("tree_test.csv",index=False,sep=',')
error_tree

