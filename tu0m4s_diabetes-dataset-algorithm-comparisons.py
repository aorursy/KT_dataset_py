# Compare algorithms
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Load data
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('../input/diabetes.csv', names=names)
data = data.drop([0])
data = data.reset_index()
data = data.drop(columns=['index'])
array = data.values
X = array[:,0:8]
Y = array[:,8]
data.head()
# Prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# Evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
# Boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison for Diabetes dataset')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xlabel('Algorithm')
ax.set_ylabel('Model accuracy')
ax.set_xticklabels(names)
pyplot.show()
# Logistic regression and linear discriminant analysis seem most promising algorithms for this particular problem, and thus worthy of further study.