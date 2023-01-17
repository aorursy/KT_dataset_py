import numpy as np
import pandas as pd

from shutil import copyfile
copyfile(src="../input/visuals/visuals.py", dst="../working/visuals.py")
import visuals as vs
# read data from adult cesnsus income
data = pd.read_csv("../input/adult-census-income/adult.csv")
# show data info
from IPython.display import display
display(data.head(5))
# get features and labels
income = data["income"]
features = data.drop("income", axis=1)
# show features distributions
vs.distribution(features, "capital.gain")
vs.distribution(features, "capital.loss")
# data preprocessing
# np.log(1+x) to unit distribution for data  
preprocessing_columns = ['capital.gain', 'capital.loss']
features[preprocessing_columns] = features[preprocessing_columns].apply(lambda x: np.log(x + 1))
vs.distribution(features, "capital.gain")
vs.distribution(features, "capital.loss")
# data re-scale
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
selected_columns = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
features[selected_columns] = scaler.fit_transform(features[selected_columns])

vs.distribution(features, "capital.gain")
vs.distribution(features, "capital.loss")
# encode for features with hot code
features = pd.get_dummies(features)

# turn income into number type
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
income = le.fit_transform(income)

display(features.head(5))
display(income[0:5])
# data split for train, val, test
from sklearn.model_selection import train_test_split
train_features, test_features, train_income, test_income = train_test_split(features, income, test_size=0.2, random_state=0)
train_features, val_features, train_income, val_income = train_test_split(train_features, train_income, test_size=0.2, random_state=0)
print(train_features.shape)
print(val_features.shape)
print(test_features.shape)
from sklearn.metrics import fbeta_score, accuracy_score
from time import time
beta = 0.5
# build model
def train_model(learner, train_length, train_x, train_y, val_x, val_y):
    result = {}
    start = time()
    learner.fit(train_x[:train_length], train_y[:train_length])
    end = time()
    result["train_time"] = end - start
    
    train_predictions = learner.predict(train_x)
    result["f_train"] = fbeta_score(train_y, train_predictions, beta)
    result["acc_train"] = accuracy_score(train_y, train_predictions)
    
    start = time()
    val_predictions = learner.predict(val_x)
    end = time()
    result["pred_time"] = end - start
    
    # val dict, mark them as test because of visuals.py need
    result["f_test"] = fbeta_score(val_y, val_predictions, beta)
    result["acc_test"] = accuracy_score(val_y, val_predictions)
    
    return result
# check different models
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

ab = AdaBoostClassifier(random_state=0)
knn = KNeighborsClassifier()
svc = SVC(random_state=0)

samples_1 = int(len(train_features) * 0.01)
samples_10 = int(len(train_features) * 0.1)
samples_100 = len(train_features)

results = {}
for clf in [ab, knn, svc]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_model(clf, samples, train_features, train_income, val_features, val_income)
        print ("{} trained on {} samples.".format(clf_name, samples))
vs.visualize_classification_performance(results)
# check features importance
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(train_features, train_income)
importances = rfc.feature_importances_
vs.feature_plot(importances, train_features, train_income)
# filter for 5 most important features
train_features_reduced = train_features[train_features.columns.values[(np.argsort(importances)[::-1])[:5]]]
val_features_reduced = val_features[val_features.columns.values[(np.argsort(importances)[::-1])[:5]]]
from sklearn.model_selection import GridSearchCV
from  sklearn.metrics import fbeta_score,make_scorer

learner = AdaBoostClassifier(random_state=0)
params = {"n_estimators": [10, 20, 50 ,100]}
score = make_scorer(fbeta_score, beta=0.5)
gscv = GridSearchCV(learner, params, score)
gscv.fit(train_features_reduced, train_income)

reduced_predictions = gscv.predict(val_features_reduced)
print("\n %s trained on reduced data\n------" %gscv.best_estimator_)
print("Accuracy on validation data: {:.4f}".format(accuracy_score(val_income, reduced_predictions)))
print("F-score on validation data: {:.4f}".format(fbeta_score(val_income, reduced_predictions, beta = 0.5)))
# show the performance on test data set
final_model = AdaBoostClassifier(n_estimators=100, random_state=0)
final_model.fit(train_features, train_income)
predictions = final_model.predict(test_features)
print("Accuracy on test data: {:.4f}".format(accuracy_score(test_income, predictions)))
print("F-score on test data: {:.4f}".format(fbeta_score(test_income, predictions, beta = 0.5)))
