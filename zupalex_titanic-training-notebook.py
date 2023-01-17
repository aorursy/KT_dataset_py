import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from array import *
import math
import string
import random
import sys
import os

train_ds = pd.read_csv("../input/train.csv")
test_ds = pd.read_csv("../input/test.csv")

def get_ratio_uncertainties(s1, s2):
    errs_low = array("f")
    errs_up = array("f")
    
    for i in range(0,s2.size):
        key = s2.index[i]
        
        if (key in s1) and (s1[key] > 0):
            ratio = s1[key]/s2[key]
            err_low = (ratio * np.sqrt((1/np.sqrt(s1[key]))**2+(1/np.sqrt(s2[key])**2)))
            err_up = err_low
            
            if ratio - err_low < 0:
                err_low = ratio
            
            if ratio + err_up > 1:
                err_up = 1-ratio
            
            errs_low.append(err_low)
            errs_up.append(err_up)
        else:
            errs_low.append(0)
            errs_up.append(0) #This is not correct especially for cases with low statistic but some more computation is needed for that
    
    return [errs_low, errs_up]
    
def get_autopct_with_val(vals):
    def autopct_with_val(pct):
        tot = sum(vals)
        raw_val = int(round(pct*tot/100.0))
        return "{0:2.2f}%  ({1:d})".format(pct,raw_val)
    return autopct_with_val
train_ds.Age = train_ds.Age.fillna(-1)
age_cats = pd.cut(train_ds.Age, bins=[-1.5,0,2,12,18,35,60,90], labels=["Unknown", "Babies", "Children", "Teenagers", "Young Adults", "Adults", "Seniors"])
train_ds = pd.concat([train_ds, age_cats.rename("AgeCat")], axis=1)
#--------------------------------- Age Distribution ----------------------------------------------

print("Passengers with known age: ", train_ds.Age.count())

plt.figure(figsize=(30,8))

plt.subplot2grid((1,3), (0,0))
plt.hist([train_ds.Age.dropna().values], bins=91, range=(0,90), stacked=True, color="#3333ff")
plt.hist([train_ds[train_ds.Survived == 1].Age.dropna().values], bins=91, range=(0,90), stacked=True, color="#29a04d")
plt.title("Age Distribution", size=20)
plt.xlabel("Age", size=15)
plt.ylabel("Count", size=15)
hist_legend = [Rectangle((0,0),1,1,color=c,ec="k") for c in ["#3333ff", "#29a04d"]]
hist_labels= ["Total Age Distribution", "Survived Age Distribution"]
plt.legend(hist_legend, hist_labels, loc=1, fontsize=20)

plt.subplot2grid((1,3), (0,1))

train_ds.AgeCat.value_counts().sort_index().plot.bar(width=0.2, color="#5DADE2", position=1)
train_ds[train_ds.Survived==1].AgeCat.value_counts().sort_index().plot.bar(width=0.2, color="#52BE80", position=0)
plt.title("Age Categories", size=20)
hist_legend = [Rectangle((0,0),1,1,color=c,ec="k") for c in ["#5DADE2", "#52BE80"]]
hist_labels= ["Total", "Survived"]
plt.legend(hist_legend, hist_labels, loc=1, fontsize=20)
plt.ylabel("Counts", fontsize=15)

plt.subplot2grid((1,3), (0,2))
age_cats_ratio = train_ds[train_ds.Survived==1].AgeCat.value_counts().sort_index()/train_ds.AgeCat.value_counts().sort_index()
age_cats_ratio.plot.bar(width=0.2, color="#52BE80", position=0.5, ecolor="#196F3D", yerr=get_ratio_uncertainties(train_ds[train_ds.Survived==1].AgeCat.value_counts().sort_index(),train_ds.AgeCat.value_counts().sort_index()))
plt.title("Survival Rate per Age Categories", size=20)

plt.show()
#--------------------------------- Survival by Passenger gender ----------------------------------------------
female_survival = (train_ds[train_ds.Sex == "female"].Survived.sum() / train_ds[train_ds.Sex == "female"].Survived.count())
male_survival = (train_ds[train_ds.Sex == "male"].Survived.sum() / train_ds[train_ds.Sex == "male"].Survived.count())
print("Women survival rate = ", female_survival)
print("Men survival rate = ", male_survival)
plt.figure(figsize=(15,8))

plt.subplot2grid((1,2), (0,0))
train_ds[train_ds.Sex == "female"].Survived.value_counts().plot.pie(title="Women Survival", autopct="%1.2f%%", explode=[0.05,0.05], shadow=True, colors=["#82E0AA", "#D98880"])
plt.subplot2grid((1,2), (0,1))
train_ds[train_ds.Sex == "male"].Survived.value_counts().plot.pie(title="Men Survival", autopct="%1.2f%%", explode=[0.05,0.05], shadow=True, startangle=160, colors=["#D98880", "#82E0AA"])

plt.figure(figsize=(15,8))

plt.subplot2grid((1,2), (0,0))
train_ds[train_ds.Survived == 1].Sex.value_counts().plot.pie(title="Survived", autopct="%1.2f%%", explode=[0.05,0.05], shadow=True, colors=["#D7BDE2", "#7FB3D5"])
plt.subplot2grid((1,2), (0,1))
train_ds[train_ds.Survived == 0].Sex.value_counts().plot.pie(title="Died", autopct="%1.2f%%", explode=[0.05,0.05], shadow=True, startangle=160, colors=["#7FB3D5", "#D7BDE2"])
#------------------------------ Embarked Distribution --------------------------------------------

print("Passenger embarked per port: Cherbourg ->", train_ds[train_ds.Embarked == "C"].Name.count(), " / Queenstown ->", train_ds[train_ds.Embarked == "Q"].Name.count(), " / Southampton ->", train_ds[train_ds.Embarked == "S"].Name.count())

embarked_survived = train_ds[train_ds.Survived == 1].Embarked

plt.figure(figsize=(20,8))

plt.subplot2grid((1,5), (0,0), colspan=2)
wedgs, pie_labels, autotxts = plt.pie(train_ds.Embarked.value_counts(), labels=["Southampton", "Cherbourg", "Queenstown"], shadow=True, autopct="%1.2f%%", explode=[0.02,0.02,0.02])
plt.title("Embarked At", fontsize=25)
[txt.set_fontsize(20) for txt in pie_labels]

plt.subplot2grid((1,5), (0,3), colspan=2)
bar_erry = get_ratio_uncertainties(embarked_survived.value_counts(), train_ds.Embarked.value_counts())
plt.bar(train_ds.Embarked.value_counts().index, embarked_survived.value_counts().values/train_ds.Embarked.value_counts().values, yerr=bar_erry, capsize=3, ecolor="#7f66af")
plt.xticks(train_ds.Embarked.value_counts().index, ["Southampton", "Cherbourg", "Queenstown"])
plt.ylabel("Survival ratio", fontsize=15)
plt.title("Survival rate per embarked port", fontsize=25)

plt.show()

print("Uncertainties are estimated using the statistical uncertainties (square root of the total amount of passenger in the considered category).")
print("Uncertainties are overestimated since the covariance between the survived and total is set to 0 (while it should be positive)")
#------------------------------- Survivability by Fare ------------------------------------------

print("Passengers with known fare: ", train_ds.Fare.count())
print("Max/Min fare paid:", train_ds.Fare.max(), "(", train_ds[train_ds.Fare == train_ds.Fare.max()].Name.count(), " passengers) / ", train_ds.Fare.min(), "(", train_ds[train_ds.Fare == train_ds.Fare.max()].Name.count(), " passengers)")

plt.figure(figsize=(15,10))

plt.subplot2grid((6,3),(1,0), colspan=2, rowspan=3)
ports_name = ["Southampton", "Cherbourg", "Queenstown"]
plt.bar(ports_name, [train_ds[train_ds.Embarked=="S"].Fare.mean(), train_ds[train_ds.Embarked=="C"].Fare.mean(), train_ds[train_ds.Embarked=="Q"].Fare.mean()])
plt.title("Average Fare per Embarked port", fontsize=20)
plt.ylabel("Ticket price", fontsize=12)

emb_cherb = train_ds[train_ds.Embarked=="C"]
emb_south = train_ds[train_ds.Embarked=="S"]
emb_queen = train_ds[train_ds.Embarked=="Q"]

pie_explodes = [0.02, 0.02, 0.02]

plt.subplot2grid((6,3),(0,2), rowspan=2)
wedgs, pie_labels, autotxts = plt.pie(emb_cherb.Pclass.value_counts(), labels=[("Class "+str(pclass)) for pclass in emb_cherb.Pclass.value_counts().index], shadow=True, autopct=get_autopct_with_val(emb_cherb.Pclass.value_counts()), explode=pie_explodes, colors=["#76D7C4", "#D98880", "#5DADE2"])
plt.title("Tickets Class for Cherbourg", fontsize=20)
[txt.set_fontsize(12) for txt in pie_labels]
[txt.set_fontsize(8) for txt in autotxts]

plt.subplot2grid((6,3),(2,2), rowspan=2)
wedgs, pie_labels, autotxts = plt.pie(emb_south.Pclass.value_counts(), startangle=160, labels=[("Class "+str(pclass)) for pclass in emb_south.Pclass.value_counts().index], shadow=True, autopct=get_autopct_with_val(emb_south.Pclass.value_counts()), explode=pie_explodes, colors=["#D98880", "#5DADE2", "#76D7C4"])
plt.title("Tickets Class for Southampton", fontsize=20)
[txt.set_fontsize(12) for txt in pie_labels]
[txt.set_fontsize(8) for txt in autotxts]

plt.subplot2grid((6,3),(4,2), rowspan=2)
wedgs, pie_labels, autotxts = plt.pie(emb_queen.Pclass.value_counts(), pctdistance=0.8, startangle=125, labels=[("Class "+str(pclass)) for pclass in emb_queen.Pclass.value_counts().index], shadow=True, autopct=get_autopct_with_val(emb_queen.Pclass.value_counts()), explode=pie_explodes, colors=["#D98880", "#5DADE2", "#76D7C4"])
plt.title("Tickets Class for Queenstown", fontsize=20)
[txt.set_fontsize(12) for txt in pie_labels]
[txt.set_fontsize(8) for txt in autotxts]

plt.show()
plt.figure(figsize=(10,7))
plt.hist([train_ds.Fare.dropna().values], range=(0,math.ceil(train_ds.Fare.max())), bins=math.ceil(train_ds.Fare.max())+1, color="#3333ff")
plt.hist([train_ds[train_ds.Survived == 1].Age.dropna().values], range=(0,math.ceil(train_ds.Fare.max())), bins=math.ceil(train_ds.Fare.max())+1, stacked=True, color="#29a04d")
plt.title("Fare Distribution", size=20)
plt.xlabel("Fare", size=15)
plt.xticks(np.arange(0, math.ceil(train_ds.Fare.max()+1), 25))
plt.ylabel("Count", size=15)
plt.yscale("log", nonposy="clip")
hist_legend = [Rectangle((0,0),1,1,color=c,ec="k") for c in ["#3333ff", "#29a04d"]]
hist_labels= ["Total Fare Distribution", "Survived Fare Distribution"]
plt.legend(hist_legend, hist_labels, loc=1, fontsize=20)
plt.show()

plt.figure(figsize=(16,25))

tot_pclass_ds = train_ds.Pclass.value_counts().sort_index()
survived_pclass_ds = train_ds[train_ds.Survived==1].Pclass.value_counts().sort_index()

plt.subplot2grid((4,4), (0,0), colspan=2)
wedgs, pie_labels, autotxts = plt.pie(tot_pclass_ds, colors=["#82E0AA", "#5DADE2", "#F7DC6F"], autopct=get_autopct_with_val(tot_pclass_ds), shadow=True, labels=[("Class "+str(pclass)) for pclass in tot_pclass_ds.index])
plt.title("Passenger Class", fontsize=25)
[txt.set_fontsize(15) for txt in pie_labels]
[txt.set_fontsize(10) for txt in autotxts]

plt.subplot2grid((4,4), (0,2), colspan=2)
wedgs, pie_labels, autotxts = plt.pie(survived_pclass_ds, colors=["#82E0AA", "#5DADE2", "#F7DC6F"], autopct=get_autopct_with_val(survived_pclass_ds), shadow=True, labels=[("Class "+str(pclass)) for pclass in survived_pclass_ds.index])
plt.title("Survived Passenger Class", fontsize=25)
[txt.set_fontsize(15) for txt in pie_labels]
[txt.set_fontsize(10) for txt in autotxts]

plt.subplot2grid((4,4), (1,1), colspan=2)
plt.bar(tot_pclass_ds.index, tot_pclass_ds.values, color="#5DADE2", align="edge", width = 0.2, linewidth=1, edgecolor=["#2874A6" for _ in tot_pclass_ds.index], tick_label=["1st Class", "2nd Class", "3rd Class"])
plt.bar(survived_pclass_ds.index - 0.2, survived_pclass_ds.values, color="#52BE80", align="edge", linewidth=1, edgecolor=["#1E8449"  for _ in survived_pclass_ds.index], width = 0.2)
plt.title("Survival per Passenger Class", fontsize=25)
bar_legend = [Rectangle((0,0),1,1,color=c,ec="k") for c in ["#5DADE2", "#52BE80"]]
hist_labels= ["Total", "Survived"]
plt.legend(hist_legend, hist_labels, loc=0, fontsize=20)

plt.show()

print("The survival ratio per passenger class is:")
survival_ratio_pclass = survived_pclass_ds/tot_pclass_ds
print("1st:", survival_ratio_pclass.values[0])
print("2nd:", survival_ratio_pclass.values[1])
print("3rd:", survival_ratio_pclass.values[2])
#------------------------------- Sibling/Spouse and Parent/Children ------------------------------------------

plt.figure(figsize=(20,16))

sibsp = train_ds.SibSp.value_counts().sort_index()
sibsp_survived = train_ds[train_ds.Survived==1].SibSp.value_counts().sort_index()
sibsp_ratio = sibsp_survived.fillna(0)/sibsp

parch = train_ds.Parch.value_counts().sort_index()
parch_survived = train_ds[train_ds.Survived==1].Parch.value_counts().sort_index()
parch_ratio = parch_survived.fillna(0)/parch

plt.subplot2grid((2,2), (0,0))
plt.bar(sibsp.index-0.2, sibsp.values, color="#5DADE2", align="edge", width = 0.2, linewidth=1, edgecolor=["#2874A6" for _ in sibsp.index])
plt.bar(sibsp_survived.index, sibsp_survived.values, color="#52BE80", align="edge", width = 0.2, linewidth=1, edgecolor=["#1E8449" for _ in sibsp_survived.index])
plt.title("Passengers Siblings/Spouses", fontsize=25)
plt.yscale("log", nonposy="clip")
bar_legend = [Rectangle((0,0),1,1,color=c,ec="k") for c in ["#5DADE2", "#52BE80"]]
hist_labels= ["Total", "Survived"]
plt.legend(hist_legend, hist_labels, loc=0, fontsize=20)

plt.subplot2grid((2,2), (0,1))
plt.bar(sibsp.index, sibsp_ratio.fillna(0).values, color="#52BE80", yerr=get_ratio_uncertainties(sibsp_survived, sibsp))
plt.title("Survival Ratio for #Siblings/Spouses", fontsize=25)

plt.subplot2grid((2,2), (1,0))
plt.bar(parch.index-0.2, parch.values, color="#5DADE2", align="edge", width = 0.2, linewidth=1, edgecolor=["#2874A6" for _ in parch.index])
plt.bar(parch_survived.index, parch_survived.values, color="#52BE80", align="edge", width = 0.2, linewidth=1, edgecolor=["#1E8449" for _ in parch_survived.index])
plt.title("Passengers Parents/Children", fontsize=25)
plt.yscale("log", nonposy="clip")
bar_legend = [Rectangle((0,0),1,1,color=c,ec="k") for c in ["#5DADE2", "#52BE80"]]
hist_labels= ["Total", "Survived"]
plt.legend(hist_legend, hist_labels, loc=0, fontsize=20)

plt.subplot2grid((2,2), (1,1))
plt.bar(parch.index, parch_ratio.fillna(0).values, color="#52BE80", yerr=get_ratio_uncertainties(parch_survived, parch))
plt.title("Survival Ratio for #Parents/Children", fontsize=25)

plt.show()

train_ds_clean = train_ds.drop(labels=["Name", "Ticket", "Fare", "Cabin", "Embarked", "SibSp", "Parch", "PassengerId", "Age"], axis=1)
train_ds_clean = train_ds_clean.drop(axis=0, labels=[idx for idx in train_ds[(train_ds.Age.isnull()==True) | (train_ds.Fare.isnull()==True) | (train_ds.Sex.isnull()==True)].index])

from sklearn import preprocessing

lbl_enc = preprocessing.LabelEncoder()

lbl_enc.fit(train_ds_clean.AgeCat) 
train_ds_clean.AgeCat = lbl_enc.transform(train_ds_clean.AgeCat)

lbl_enc.fit(train_ds_clean.Sex) 
train_ds_clean.Sex = lbl_enc.transform(train_ds_clean.Sex)
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(train_ds_clean.drop(labels=["Survived"], axis=1),train_ds_clean.Survived, test_size=0.8)

svc = svm.SVC()
gs_params = {"kernel":["rbf", "poly", "linear"], "C":[1, 2, 5, 10]}
clf = GridSearchCV(svc, gs_params, verbose=True)                         #Changing the parameter order result in the first parameters being the "best" ones. It means they don't influence the outcome here
clf.fit(X_train, y_train)
print(clf.best_params_)
best_est = clf.best_estimator_

best_est.fit(X_train, y_train)
predicts = best_est.predict(X_test)
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, predicts)
print("Accuracy of the prediction on the test set:", acc)
def check_pred_accuracy(train_, test_, n, algorithm_, gs_pars):
    rdm_seeds = [random.randint(0, 2**32 - 1) for i in range(0,20)]
    acc = []
    
    ests_ = []
    
    for i in range(0,n):
        X_train_, X_test_, y_train_, y_test_ = train_test_split(train_, test_, test_size=0.8, random_state=rdm_seeds[i])
        
        alg = algorithm_()
        clf_ = GridSearchCV(alg, gs_pars)
        clf_.fit(X_train_, y_train_)
        
        clf_.best_estimator_.fit(X_train_, y_train_)
        pred_ = clf_.best_estimator_.predict(X_test_)

        acc.append(accuracy_score(y_test_, pred_))
        ests_.append(clf_.best_estimator_)

    print("Average accuracy on {ntests} randomly selected train/test sets: {avg:1.2f}%".format(ntests=n, avg=sum(acc)/len(acc)*100))
    return ests_

def get_features_importance(best_ests_, attr_):
    if len(best_ests_) == 0:
        print("ERROR: first argument is emtpy")
        return
    
    feats_ = [0 for _ in range(0,len(getattr(best_ests_[0], attr_)))]
    
    for be in best_ests_:
        coefs_ = getattr(be, attr_)
        for i in range(0, len(coefs_)):
            feats_[i] += coefs_[i]
    
    for i in range(0, len(feats_)):
        feats_[i] /= len(best_ests_)
        
    return feats_

# Switch to linear here as otherwise it's hard to tell which feature is important or not
best_est = check_pred_accuracy(train_ds_clean.drop(labels=["Survived"], axis=1),train_ds_clean.Survived, 20, svm.SVC, {"kernel":["linear"], "C":[1, 2, 5, 10]})

print("Estimator Coeffs:", get_features_importance(best_est, "coef_"))
from sklearn.ensemble import RandomForestClassifier

best_est = check_pred_accuracy(train_ds_clean.drop(labels=["Survived"], axis=1),train_ds_clean.Survived, 20, RandomForestClassifier, {"max_depth":[None, 1, 2, 5, 10], "criterion":["entropy", "gini"]})

print("Features Importance:", get_features_importance(best_est, "feature_importances_"))
from sklearn.ensemble import GradientBoostingClassifier

best_est = check_pred_accuracy(train_ds_clean.drop(labels=["Survived"], axis=1),train_ds_clean.Survived, 20, GradientBoostingClassifier, {"learning_rate":[0.01, 0.05, 0.1], "max_depth": [1, 2, 5, 10]})

print("Features Importance:", get_features_importance(best_est, "feature_importances_"))
train_ds_clean2 = pd.concat([train_ds_clean, train_ds.SibSp], axis=1)

print("SVC from sklearn.svm")
best_est = check_pred_accuracy(train_ds_clean2.drop(labels=["Survived"], axis=1),train_ds_clean2.Survived, 20, svm.SVC, {"kernel":["linear"], "C":[1, 2, 5, 10]})
print("Estimator Coeffs:", get_features_importance(best_est, "coef_"))

print("RandomForestClassifier")
best_est = check_pred_accuracy(train_ds_clean2.drop(labels=["Survived"], axis=1),train_ds_clean2.Survived, 20, RandomForestClassifier, {"max_depth":[None, 1, 2, 5, 10], "criterion":["entropy", "gini"]})
print("Features Importances:", get_features_importance(best_est, "feature_importances_"))

print("GradientBoostingClassifier")
best_est = check_pred_accuracy(train_ds_clean2.drop(labels=["Survived"], axis=1),train_ds_clean2.Survived, 20, GradientBoostingClassifier, {"learning_rate":[0.01, 0.05, 0.1], "max_depth": [1, 2, 5, 10]})
print("Features Importances:", get_features_importance(best_est, "feature_importances_"))
train_ds_clean3 = pd.concat([train_ds_clean, train_ds.Parch], axis=1)

print("SVC from sklearn.svm")
best_est = check_pred_accuracy(train_ds_clean3.drop(labels=["Survived"], axis=1),train_ds_clean3.Survived, 20, svm.SVC, {"kernel":["linear"], "C":[1, 2, 5, 10]})
print("Estimator Coeffs:", get_features_importance(best_est, "coef_"))

print("RandomForestClassifier")
best_est = check_pred_accuracy(train_ds_clean3.drop(labels=["Survived"], axis=1),train_ds_clean3.Survived, 20, RandomForestClassifier, {"max_depth":[None, 1, 2, 5, 10], "criterion":["entropy", "gini"]})
print("Features Importances:", get_features_importance(best_est, "feature_importances_"))

print("GradientBoostingClassifier")
best_est = check_pred_accuracy(train_ds_clean3.drop(labels=["Survived"], axis=1),train_ds_clean3.Survived, 20, GradientBoostingClassifier, {"learning_rate":[0.01, 0.05, 0.1], "max_depth": [1, 2, 5, 10]})
print("Features Importances:", get_features_importance(best_est, "feature_importances_"))
train_ds_clean4 = pd.concat([train_ds_clean, train_ds.Parch, train_ds.SibSp], axis=1)

print("SVC from sklearn.svm")
best_est = check_pred_accuracy(train_ds_clean4.drop(labels=["Survived"], axis=1),train_ds_clean4.Survived, 20, svm.SVC, {"kernel":["linear"], "C":[1, 2, 5, 10]})
print("Estimator Coeffs:", get_features_importance(best_est, "coef_"))

print("RandomForestClassifier")
best_est = check_pred_accuracy(train_ds_clean4.drop(labels=["Survived"], axis=1),train_ds_clean4.Survived, 20, RandomForestClassifier, {"max_depth":[None, 1, 2, 5, 10], "criterion":["entropy", "gini"]})
print("Features Importances:", get_features_importance(best_est, "feature_importances_"))

print("GradientBoostingClassifier")
best_est = check_pred_accuracy(train_ds_clean4.drop(labels=["Survived"], axis=1),train_ds_clean4.Survived, 20, GradientBoostingClassifier, {"learning_rate":[0.01, 0.05, 0.1], "max_depth": [1, 2, 5, 10]})
print("Features Importances:", get_features_importance(best_est, "feature_importances_"))
test_ds.Age = test_ds.Age.fillna(-1)
test_age_cats = pd.cut(test_ds.Age, bins=[-1.5,0,2,12,18,35,60,90], labels=["Unknown", "Babies", "Children", "Teenagers", "Young Adults", "Adults", "Seniors"])
test_ds = pd.concat([test_ds, test_age_cats.rename("AgeCat")], axis=1)

test_ds_clean = test_ds.drop(labels=["Name", "Ticket", "Fare", "Cabin", "Embarked", "PassengerId", "Age"], axis=1)
test_ds_clean = test_ds_clean.drop(axis=0, labels=[idx for idx in train_ds[(train_ds.Age.isnull()==True) | (train_ds.Fare.isnull()==True) | (train_ds.Sex.isnull()==True)].index])

lbl_enc.fit(test_ds_clean.AgeCat) 
test_ds_clean.AgeCat = lbl_enc.transform(test_ds_clean.AgeCat)

lbl_enc.fit(test_ds_clean.Sex) 
test_ds_clean.Sex = lbl_enc.transform(test_ds_clean.Sex)

predicts = best_est[0].predict(test_ds_clean)

pred_list = pd.concat([test_ds.PassengerId, pd.DataFrame({"Survived": predicts})], axis=1)
pred_list.to_csv(path_or_buf="attempt1.csv", index=False)