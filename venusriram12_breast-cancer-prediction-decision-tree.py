import pandas as pd
data=pd.read_csv("../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv", delimiter=",")
data.head()
data.shape
data["diagnosis"].value_counts()
data.info()
Y=data["diagnosis"]
X=data.drop("diagnosis",axis=1)
X.shape
X.corr()
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(X.corr(),annot=True)
sns.pairplot(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
%time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint as sp_randint

depths=[1,5,50,100,150,250]
samples=[5, 10,50, 100,250, 500]

clf = DecisionTreeClassifier()

params = {'max_depth' : depths,
         "min_samples_split":samples}

grid = GridSearchCV(estimator = clf,param_grid=params ,cv = 3,n_jobs = 3,scoring='roc_auc')
grid.fit(X_train, y_train)
print("best depth = ", grid.best_params_)
print("AUC value on train data = ", grid.best_score_*100)
a = grid.best_params_
optimal_depth = a.get('depth')
optimal_split=a.get("min_samples_split")
clf = DecisionTreeClassifier(max_depth=optimal_depth,min_samples_split=optimal_split) 

clf.fit(X_train,y_train)

pred = clf.predict(X_test)


from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
import numpy as np
scores = grid.cv_results_['mean_test_score'].reshape(len(samples),len(depths))

plt.figure(figsize=(16, 12))
sns.heatmap(scores, annot=True, cmap=plt.cm.hot, fmt=".3f", xticklabels=samples, yticklabels=depths)
plt.xlabel('min_samples_split')
plt.ylabel('max_depth')
plt.xticks(np.arange(len(samples)), samples)
plt.yticks(np.arange(len(depths)), depths)
plt.title('Grid Search AUC Score')
plt.show()
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
acc1 = accuracy_score(y_test, pred) * 100
pre1 = precision_score(y_test, pred) * 100
rec1 = recall_score(y_test, pred) * 100
f11 = f1_score(y_test, pred) * 100
print('\nAccuracy=%f%%' % (acc1))
print('\nprecision=%f%%' % (pre1))
print('\nrecall=%f%%' % (rec1))
print('\nF1-Score=%f%%' % (f11))
cm = confusion_matrix(y_test,pred)
sns.heatmap(cm, annot=True,fmt='d')
plt.title('Confusion Matrix for BoW')
plt.show()
from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier(max_depth=2,min_samples_split=3)
dtree.fit(X_train,y_train)
from matplotlib import pyplot
importance =dtree.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
