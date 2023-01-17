import numpy as np

import pandas as pd

import seaborn as sns

import xgboost as xgb

from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedKFold, train_test_split, KFold

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Import data

data = pd.read_csv("../input/zoo.csv",sep=",")

data.drop('animal_name',axis=1,inplace=True)
# Data and Target

y = data['class_type']

x = data.drop('class_type',axis = 1)



# Functions

def score(clf, train_np, random_state, folds, target):

    kf = StratifiedKFold(n_splits = folds, shuffle = False, random_state = random_state)

    list_perf = []

    for itrain, itest in kf.split(train_np,target):

        Xtr, Xte = train_np[itrain], train_np[itest]

        ytr, yte = target[itrain], target[itest]

        clf.fit(Xtr, ytr.ravel())

        pred = pd.DataFrame(clf.predict(Xte)) 

        list_perf.append(metrics.accuracy_score(yte,pred))

    return list_perf



def score_seed(clf,nbseed,listout,printoutput):

    listout = []

    for i in range(nbseed):

        list1 = score(clf,np.array(x),random_state = i, folds = 4, target = y )

        listout.append(list1)

    print(printoutput) 

    return listout
# Modelisation

# 10 seed differents. for each classifications we'll take 4 folds

rf = RandomForestClassifier(n_estimators=200,n_jobs=-1)

listrf = []

listrf = score_seed(rf,10,listrf,"RF completed")



et = ExtraTreesClassifier(n_estimators=200,n_jobs=-1)

listet = []

listet = score_seed(et,10,listet,"ET completed")



gb = GradientBoostingClassifier(n_estimators=200,learning_rate = 0.1)

listgb = []

listgb = score_seed(gb,10,listgb,"GB completed")



gnb = GaussianNB()

listgnb = []

listgnb = score_seed(gnb,10,listgnb,"GNB completed")



kn = KNeighborsClassifier(n_neighbors=1)

listkn = []

listkn = score_seed(kn,10,listkn,"KN completed")



clf1 = RandomForestClassifier(n_estimators=200,n_jobs=-1)

clf2 = ExtraTreesClassifier(n_estimators=200,n_jobs=-1)

clf3 = GradientBoostingClassifier(n_estimators=200,learning_rate = 0.1)

clf4 = GaussianNB()

vc = VotingClassifier(estimators=[('rf', clf1), ('et', clf2), ('gb', clf3),('gnb', clf4)], voting='soft')

listvc = []

listvc = score_seed(vc,10,listvc,"VC completed")



listxgb = []

xgbC =  xgb.XGBClassifier(n_estimators=200)

listxgb = score_seed(xgbC,5,listxgb,"XGB completed")



print("RF perf:" ,   np.mean(listrf))

print("ET perf:" ,   np.mean(listet))

print("GB perf:" ,   np.mean(listgb))

print("GNB perf:",   np.mean(listgnb))

print("VC perf:" ,   np.mean(listvc))

print("KN perf:" ,   np.mean(listkn))

print("XGB perf:",   np.mean(listxgb))
list_model = ["Random Forest", "Extra Trees", "Gradient Boosting", 

"Gausian NB", "Voting", "KNN", "XGBoost"]



list_pref  = [np.mean(listrf), np.mean(listet), np.mean(listgb), np.mean(listgnb), np.mean(listvc), np.mean(listkn), np.mean(listxgb)]
print(list_pref)
# Performance of models

fg,ax = plt.subplots(nrows=1,ncols=1,figsize=(6, 2.5))

sns.barplot(list_model, list_pref,palette="RdBu")

plt.ylim(0.93, 0.97)

ax.set_ylabel("Performance")

ax.set_xlabel("Name")

ax.set_xticklabels(list_model,rotation=35)

plt.title('Battle of Algorithms')
# Feature importance for RF, ET and XGB