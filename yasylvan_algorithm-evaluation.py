#导入类库
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
%matplotlib inline
dataset = pd.read_csv('../input/Iris.csv')
dataset.shape
dataset.head()
dataset.drop('Id',axis=1,inplace=True)
dataset.describe()
dataset.groupby('Species').size()
dataset.plot(kind = 'box',subplots = True,sharex=False,sharey=False);
dataset.hist();
pd.scatter_matrix(dataset);
X=dataset.values[:,0:4]
y=dataset.values[:,4]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 7)
import warnings
warnings.filterwarnings("ignore")
#算法审查
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()
#评估算法
results = []
for key in models:
    kfold = KFold(n_splits=10,random_state = 7)
    cv_results = cross_val_score(models[key],X_train ,y_train,cv = kfold,scoring='accuracy')
    results.append(cv_results)
    print('%s:%f(%f)' %(key,cv_results.mean(),cv_results.std()))
#箱线图比较算法
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys());
svm=SVC()
svm.fit(X=X_train,y = y_train)
predictions = svm.predict(X_test)
print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
