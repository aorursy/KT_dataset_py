import numpy as np

import pandas as pd 

import matplotlib.pyplot as mp

from sklearn.model_selection import cross_val_score

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

pd.set_option('max_columns', 1000)

pd.set_option('max_rows', 10)
data= pd.read_csv('../input/clustering_data.csv',encoding='latin1')
y = data.IS_BOUND

x = data.drop('IS_BOUND', axis=1)



from sklearn.preprocessing import LabelEncoder

L = LabelEncoder()

for i in x.columns:

    if x[i].dtypes == 'object':

        x[i] = L.fit_transform(x[i].astype(str))

        

from scipy.stats import pearsonr

for I in x.columns:

   print(I,pearsonr(x[i],y))



from sklearn.feature_selection import chi2, SelectKBest

a = SelectKBest(score_func=chi2,k=10)

x1 = a.fit_transform(x, y)

c=['VEHICLEMAKE','VEHICLEMODEL','ANNUAL_KM','COMMUTE_DISTANCE','Value','POSTAL_CODE','AREA_CODE','MULTI_PRODUCT','age','province']

x1=pd.DataFrame(x1,columns=c)



from sklearn.preprocessing import StandardScaler

a = StandardScaler ()

x2= a.fit_transform(x1)

x2=pd.DataFrame(x2,columns=x1.columns)



from sklearn.preprocessing import MinMaxScaler

a = MinMaxScaler ()

x3= a.fit_transform(x2)

x3=pd.DataFrame(x3,columns=x2.columns)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x2, y, test_size=0.2, shuffle=True)

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(class_weight='balanced')

parameters = {'max_depth':range(2,6) , 'min_samples_leaf':range(20,50,10),'max_leaf_nodes':range(10,15)}

from sklearn.model_selection import GridSearchCV

model = GridSearchCV(tree, parameters, scoring='accuracy', return_train_score=True)

model.fit(x_train, y_train)

from sklearn import metrics

y_pred = model.predict(x_test)

print(model.score(x_test, y_test))

print(metrics.confusion_matrix(y_test,y_pred))

print(metrics.mean_squared_error(y_test,y_pred))
from sklearn.model_selection import cross_val_score

from sklearn.datasets import make_blobs

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(model.score(x_test, y_test))

print(metrics.confusion_matrix(y_test,y_pred))

print(metrics.mean_squared_error(y_test,y_pred))
importance = model.feature_importances_

headers = x2.columns



import matplotlib.pyplot as mp

mp.figure('Feature Importance', facecolor='lightgray',figsize=(25,7))

mp.title('Car Insurance')

mp.ylabel('Feature Importance')

mp.grid(linestyle=":")

sorted_indexes = importance.argsort()[::-1]

x = np.arange(headers.size)

mp.bar(x, importance[sorted_indexes], 0.5, color='dodgerblue')

mp.xticks(x, headers[sorted_indexes])

mp.legend()
model= RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=2, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,

            oob_score=False, random_state=0, verbose=0, warm_start=False)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(model.score(x_test, y_test)) 

print(metrics.confusion_matrix(y_test,y_pred))

print(metrics.mean_squared_error(y_test,y_pred))
importance = model.feature_importances_

headers = x2.columns



import matplotlib.pyplot as mp

mp.figure('Feature Importance', facecolor='lightgray',figsize=(25,7))

mp.title('Car Insurance')

mp.ylabel('Feature Importance')

mp.grid(linestyle=":")

sorted_indexes = importance.argsort()[::-1]

x = np.arange(headers.size)

mp.bar(x, importance[sorted_indexes], 0.5, color='g')

mp.xticks(x, headers[sorted_indexes])

mp.legend()

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(n_estimators=100)

scores = cross_val_score(model,x_train, y_train)

print('accuracy：',scores.mean())
from sklearn import datasets

from sklearn.ensemble import GradientBoostingClassifier

 

clf = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=1, random_state=0)

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.utils import shuffle



model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01,max_depth=4,min_samples_split=2,loss='ls')

model.fit(x_train, y_train)

print('MSE：',mean_squared_error(y_test, model.predict(x_test)))



mp.plot(np.arange(500), model.train_score_, 'b-') 

mp.show()
importance = model.feature_importances_

headers = x2.columns



import matplotlib.pyplot as mp

mp.figure('Feature Importance', facecolor='lightgray',figsize=(25,7))

mp.title('Car Insurance')

mp.ylabel('Feature Importance')

mp.grid(linestyle=":")

sorted_indexes = importance.argsort()[::-1]

x = np.arange(headers.size)

mp.bar(x, importance[sorted_indexes], 0.5, color='k')

mp.xticks(x, headers[sorted_indexes])

mp.legend()
model = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(model.score(x_test, y_test)) 

print(metrics.confusion_matrix(y_test,y_pred))

print(metrics.mean_squared_error(y_test,y_pred))
importance = model.feature_importances_

headers = x2.columns



import matplotlib.pyplot as mp

mp.figure('Feature Importance', facecolor='lightgray',figsize=(25,7))

mp.title('Car Insurance')

mp.ylabel('Feature Importance')

mp.grid(linestyle=":")

sorted_indexes = importance.argsort()[::-1]

x = np.arange(headers.size)

mp.bar(x, importance[sorted_indexes], 0.5, color='r')

mp.xticks(x, headers[sorted_indexes])

mp.legend()
from sklearn.linear_model import LogisticRegression  

model= LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, 

    fit_intercept=True, intercept_scaling=1, class_weight=None, 

    random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', 

    verbose=0, warm_start=False, n_jobs=1)  

model.fit(x_train, y_train)  

y_pred = model.predict(x_test)

print(model.score(x_test, y_test)) 

print(metrics.confusion_matrix(y_test,y_pred))

print(metrics.mean_squared_error(y_test,y_pred))
from sklearn.neural_network import MLPClassifier

model=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(13,13,13),max_iter=500,random_state=1)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(model.score(x_test, y_test)) 

print(metrics.confusion_matrix(y_test,y_pred))

print(metrics.mean_squared_error(y_test,y_pred))
y = data.IS_BOUND

from sklearn.preprocessing import MinMaxScaler

a = MinMaxScaler()

x3= a.fit_transform(x2)

x3=pd.DataFrame(x3,columns=x2.columns)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x3, y, test_size=0.2, shuffle=True)



from sklearn.naive_bayes import GaussianNB 

model = GaussianNB(priors=None)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(model.score(x_test, y_test)) 

print(metrics.confusion_matrix(y_test,y_pred))

print(metrics.mean_squared_error(y_test,y_pred))
y = data.IS_BOUND

from sklearn.preprocessing import MinMaxScaler

a = MinMaxScaler()

x3= a.fit_transform(x2)

x3=pd.DataFrame(x3,columns=x2.columns)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x3, y, test_size=0.2, shuffle=True)



from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=1.0,fit_prior=True,class_prior=None)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(model.score(x_test, y_test)) 

print(metrics.confusion_matrix(y_test,y_pred))

print(metrics.mean_squared_error(y_test,y_pred))
def main():

    from sklearn.neighbors import KNeighborsClassifier

    best_k=-1

    best_score=0

    for i in range(1,30):

        knn_clf=KNeighborsClassifier(n_neighbors=i)

        knn_clf.fit(x_train,y_train)

        scores=knn_clf.score(x_test,y_test)

        if scores>best_score:

            best_score=scores

            best_k=i

    print('bset k:%d,best score:%.4f'%(best_k,best_score))

if __name__ == '__main__':

    main()
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 22)

model.fit(x_train, y_train)

print(model.score(x_test, y_test)) 

print(metrics.mean_squared_error(y_test,y_pred))