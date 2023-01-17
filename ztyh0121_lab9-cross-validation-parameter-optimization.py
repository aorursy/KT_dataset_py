import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
iris = datasets.load_iris()
iris.data.shape, iris.target.shape
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test) 
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
cross_val_score(clf, iris.data, iris.target, cv=cv)
from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
clf.score(X_test_transformed, y_test)  
from sklearn.pipeline import make_pipeline
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
cross_val_score(clf, iris.data, iris.target, cv=cv)
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)
print(metrics.accuracy_score(iris.target, predicted))
predicted
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from mlxtend.plotting import plot_decision_regions 
from matplotlib.colors import ListedColormap
np.random.seed(0)
X_xor=np.random.randn(200,2) 
y_xor=np.logical_xor(X_xor[:,0]>0, X_xor[:,1]>0)
print(y_xor[:3])
y_xor=np.where(y_xor,1,0)
pd.DataFrame(y_xor)
#%matplotlib inline
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==0, 0], X_xor[y_xor==0, 1], c='r', marker='s', label='0')
plt.legend(loc='best')
plt.show()
from sklearn import grid_search
X_std=X_xor
z=y_xor
X_train, X_test, train_label, test_label=model_selection.train_test_split(X_std,z, test_size=0.1, random_state=0)
clf=svm.SVC(class_weight='balanced', random_state=0)
param_range=[0.01, 0.1, 1.0]
param_grid=[{'C':param_range,'kernel':['rbf', 'linear'], 'gamma':param_range}]
gs=model_selection.GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
gs=gs.fit(X_train,train_label)
print(gs.best_score_)
print(gs.best_params_)
clf=gs.best_estimator_
pred=clf.predict(X_test)
ac_score=metrics.accuracy_score(test_label,pred)
print(ac_score)
cnfmat=metrics.confusion_matrix(y_true=test_label,y_pred=pred )
print(cnfmat)
report=metrics.classification_report(y_true=test_label,y_pred=pred )
print(report)
X_train_plot=np.vstack(X_train)
train_label_plot=np.hstack(train_label)
X_test_plot=np.vstack(X_test)
test_label_plot=np.hstack(test_label)
plot_decision_regions(X_test_plot, test_label_plot, clf=clf, res=0.01, legend=2)
plt.show()