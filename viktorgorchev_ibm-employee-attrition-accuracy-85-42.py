%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve, auc

from sklearn.cross_validation import KFold



from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier



from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier
resource_location = '../input/WA_Fn-UseC_-HR-Employee-Attrition.csv'

data = pd.read_csv(resource_location)
print("Rows count: ", data.shape[0])

print("Columns count: ", data.shape[1])
data.head(3)
data.tail(3)
print('NaN values in data:')

data.apply(lambda x: sum(x.isnull()),axis=0)  
data.dtypes
cols = data.columns

numeric_cols = data._get_numeric_data().columns

categirical_cols = cols.drop(numeric_cols.tolist())
separator = "; "

print("Numeric data:\n", separator.join(numeric_cols))

print("Categorical data:\n", separator.join(categirical_cols))
data[numeric_cols].describe()
del data["EmployeeCount"]

del data["EmployeeNumber"]

del data["StandardHours"]
for col in categirical_cols:

    print(data[col].value_counts())
data["Over18"].value_counts()
del data["Over18"]
data = pd.concat([data, pd.get_dummies(data[["Gender", "OverTime", "Attrition"]], drop_first=True)], axis=1)



del data["Gender"]

del data["OverTime"]

del data["Attrition"]
data.head(3)
sns.set(style="whitegrid", font_scale=1.3)

sns.countplot(x="Attrition_Yes", data=data, palette="hls")

sns.plt.title("Attrition Counts")

sns.plt.xlabel("Attrition (No = 0, Yes = 1)")

plt.show()
sns.set(style="whitegrid", font_scale=1)

plt.figure(figsize=(16,16))

corr = round(data.corr(),2)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, annot=True, cmap="RdBu", mask=mask, )

plt.title("Correlation between features", fontdict={"fontsize":20})

plt.show()
extract_cols  = ["Age",

                 "DistanceFromHome",

                 "JobInvolvement",

                 "JobLevel", 

                 "MonthlyIncome",

                 "StockOptionLevel",

                 "TotalWorkingYears", 

                 "YearsAtCompany", 

                 "YearsInCurrentRole",

                 "YearsWithCurrManager",

                 "YearsSinceLastPromotion", 

                 "OverTime_Yes", 

                 "Attrition_Yes"]



sns.set(style="whitegrid", font_scale=1.2)

plt.figure(figsize=(10,7))

corr = round(data[extract_cols].corr(),2)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, annot=True, cmap="RdBu", vmin=-1, vmax=1, mask=mask)

plt.title("Correlation between important features and attrition", fontdict={"fontsize":20})

plt.show()
most_relevant = ["JobLevel", 

                 "MonthlyIncome",

                 "StockOptionLevel",

                 "TotalWorkingYears", 

                 "YearsAtCompany", 

                 "YearsInCurrentRole",

                 "YearsWithCurrManager",

                 "OverTime_Yes"]



for col in most_relevant:    

    sns.factorplot(x="Attrition_Yes", y=col, data=data, kind="bar");

    sns.plt.xlabel("Attrition (No = 0, Yes = 1)")

    plt.title(col + " / Attrition")

    plt.show()
ibm_data = data[["Age",

                 "JobLevel", 

                 "MonthlyIncome",

                 "StockOptionLevel",

                 "TotalWorkingYears", 

                 "YearsAtCompany", 

                 "YearsInCurrentRole",

                 "YearsWithCurrManager",

                 "OverTime_Yes", 

                 "Attrition_Yes"]]
ibm_data.dtypes
def print_cross_validation_score(model, attributes, labels, n_folds):  

    kf = KFold(attributes.shape[0], n_folds=n_folds)

    error = []

    for train, test in kf:

        train_predictors = (attributes.iloc[train,:])

        train_target = labels.iloc[train]

        

        model.fit(train_predictors, train_target)

        

        error.append(model.score(attributes.iloc[test,:], labels.iloc[test]))

        

    print("Cross-Validation scores: ", error)

    

    print(

        "\nCross-Validation mean score : %s" % "{0:.3%}".format(np.mean(error)), 

        "(standard deviation: %s)" % "{0:.3%}".format(np.array(error).std())

    ) 
def print_clf_quality(labels_test, predicted):    

    accuracy = accuracy_score(labels_test, predicted)

    precision = precision_score(labels_test, predicted, average="weighted")

    recall = recall_score(labels_test, predicted, average="weighted")

    f1 = f1_score(labels_test, predicted, average="weighted")



    print("accuracy: ", accuracy)

    print("precision: ", precision)

    print("recall: ", recall)

    print("f1: ", f1)



    print("\nConfusion matrix :")

    print(confusion_matrix(labels_test, predicted))   
def print_roc_curve(y_score, y_test):    

    n_classes = 1

    

    fpr = dict()

    tpr = dict()

    roc_auc = dict()

    for i in range(n_classes):

        fpr[i], tpr[i], _ = roc_curve(y_test, y_score)

        roc_auc[i] = auc(fpr[i], tpr[i])

   

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())

    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    

    plt.figure()

    plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC curve')

    plt.legend(loc="lower right")

    plt.show()       
cols = ibm_data.columns.drop("Attrition_Yes")

attributes = ibm_data[cols]

labels = ibm_data["Attrition_Yes"]



features_train, features_test, labels_train, labels_test = train_test_split(attributes, labels, train_size=0.7, stratify=labels)
svm = SVC(kernel = "linear")
n_folds = 5

print_cross_validation_score(svm, features_train, labels_train, n_folds)  
svm.fit(features_train , labels_train)
predicted = svm.predict(features_test)



print_clf_quality(labels_test, predicted)
labels_score = svm.decision_function(features_test)



print_roc_curve(labels_score, labels_test)
svm_search = SVC(kernel = "linear")
#The code below code is not active because the grid search takes more than 1200 seconds and

#kaggle kills the kernel!

#The best estimator from the grid search is C=1000.





#params_svm = {"C": [0.001, 0.01, 0.1, 10, 100, 1000]}

#folds = 5

#search_svm = GridSearchCV(svm_search, params_svm, cv = folds)

#search_svm.fit(features_train , labels_train)



#print(search_svm.best_estimator_)
svm_best = SVC(C = 1000, kernel = "linear")
n_folds = 5

print_cross_validation_score(svm_best, features_train, labels_train, n_folds)  
svm_best.fit(features_train , labels_train)
predicted = svm_best.predict(features_test)



print_clf_quality(labels_test, predicted)
labels_score = svm_best.decision_function(features_test)



print_roc_curve(labels_score, labels_test)
kt = SVC(kernel = "rbf")
n_folds = 5

print_cross_validation_score(kt, features_train, labels_train, n_folds)  
kt.fit(features_train , labels_train)
predicted = kt.predict(features_test)



print_clf_quality(labels_test, predicted)
labels_score = kt.decision_function(features_test)



print_roc_curve(labels_score, labels_test)
kt_search = SVC(kernel = "rbf")
params_kt = {"C": [0.001, 0.01, 100, 1000], "gamma": [0.00001, 10]}

folds_kt = 5
search_kt = GridSearchCV(kt_search, params_kt, cv = folds_kt)

search_kt.fit(features_train , labels_train)
print(search_kt.best_estimator_)
kt_best = SVC(C = 0.001, gamma = 0.00001, kernel = "rbf")
n_folds = 5

print_cross_validation_score(kt_best, features_train, labels_train, n_folds) 
kt_best.fit(features_train , labels_train)
predicted = kt_best.predict(features_test)



print_clf_quality(labels_test, predicted)
labels_score = kt_best.decision_function(features_test)



print_roc_curve(labels_score, labels_test)
knn = KNeighborsClassifier()
n_folds = 5

print_cross_validation_score(knn, features_train, labels_train, n_folds)
knn.fit(features_train , labels_train)
predicted = knn.predict(features_test)



print_clf_quality(labels_test, predicted)
labels_score = knn.predict_proba(features_test)



print_roc_curve(labels_score[:, 1], labels_test)
knn_search = KNeighborsClassifier()
params_knn = {"n_neighbors": [2, 6, 7, 8, 9]}

folds_knn = 5

search_knn = GridSearchCV(knn_search, params_knn, cv = folds_knn)

search_knn.fit(features_train , labels_train)
print(search_knn.best_estimator_)
knn_best = KNeighborsClassifier(n_neighbors = 6)
n_folds = 5

print_cross_validation_score(knn_best, features_train, labels_train, n_folds) 
knn_best.fit(features_train , labels_train)
predicted = knn_best.predict(features_test)



print_clf_quality(labels_test, predicted)
labels_score = knn_best.predict_proba(features_test)



print_roc_curve(labels_score[:, 1], labels_test)
nn = MLPClassifier()
n_folds = 5

print_cross_validation_score(nn, features_train, labels_train, n_folds)  
nn.fit(features_train , labels_train)
predicted = nn.predict(features_test)



print_clf_quality(labels_test, predicted)
labels_score = nn.predict_proba(features_test)



print_roc_curve(labels_score[:, 1], labels_test)
nn_search = MLPClassifier()
params_nn = {

    "hidden_layer_sizes" : [(30, 30), (300, 300), (5, 10, 20, 5)], 

    "early_stopping" : [True, False],

    "alpha" : 10.0 ** - np.arange(1, 7),

    "max_iter" : [20000]

}

folds_nn = 5

search_nn = GridSearchCV(nn_search, params_nn, cv = folds_nn)

search_nn.fit(features_train , labels_train)
print(search_nn.best_estimator_)
nn_best = MLPClassifier(alpha=0.10000000000000001, early_stopping = True, hidden_layer_sizes = (300, 300))
n_folds = 5

print_cross_validation_score(nn_best, features_train, labels_train, n_folds) 
nn_best.fit(features_train , labels_train)
print("Train data score: ", nn_best.score(features_train , labels_train))

print("Test data score: ", nn_best.score(features_test, labels_test))

print()



predicted = nn_best.predict(features_test)



print_clf_quality(labels_test, predicted)
labels_score = nn_best.predict_proba(features_test)



print_roc_curve(labels_score[:, 1], labels_test)
ada_tree = DecisionTreeClassifier(max_depth=1)
ada = AdaBoostClassifier(base_estimator = ada_tree)
n_folds = 5

print_cross_validation_score(ada, features_train, labels_train, n_folds) 
ada.fit(features_train, labels_train) 
print("Train data score: ", ada.score(features_train , labels_train))

print("Test data score: ", ada.score(features_test, labels_test))

print()



predicted = ada.predict(features_test)



print_clf_quality(labels_test, predicted)
labels_score = ada.decision_function(features_test)



print_roc_curve(labels_score, labels_test)
tree = RandomForestClassifier()
ada_rfc = AdaBoostClassifier(base_estimator = tree)
n_folds = 5

print_cross_validation_score(ada_rfc, features_train, labels_train, n_folds) 
ada_rfc.fit(features_train, labels_train) 
print("Train data score: ", ada_rfc.score(features_train , labels_train))

print("Test data score: ", ada_rfc.score(features_test, labels_test))

print()



predicted = ada_rfc.predict(features_test)



print_clf_quality(labels_test, predicted)
labels_score = ada_rfc.decision_function(features_test)



print_roc_curve(labels_score, labels_test)