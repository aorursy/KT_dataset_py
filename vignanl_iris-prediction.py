import pandas as pd

import numpy as np

from sklearn import datasets



%pylab inline



pylab.rcParams['figure.figsize'] = (10,6)



iris = datasets.load_iris()



x = iris.data[:,[2,3]]

y = iris.target



iris_df = pd.DataFrame(iris.data[:,[2,3]],columns = iris.feature_names[2:])



print (iris_df.head())



print ('\n' + "The uniques labels in the data are" + str(np.unique(y)))

from sklearn.cross_validation import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)



print ("There are {} trainig samples and {} test samples".format(x_train.shape[0],x_test.shape[0]))

print()
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

sc.fit(x_train)



x_train_std = sc.transform(x_train)

x_test_std = sc.transform(x_test)



print (pd.DataFrame(x_train_std, columns = iris_df.columns).head())
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt



markers = ('s','x','o')

colors = ('red','blue','green')



cmap = ListedColormap(colors[:len(np.unique(y_test))])



for idx, cl in enumerate(np.unique(y)):

    plt.scatter(x = x[y==cl,0],y=x[y==cl,1],

               c = cmap(idx),marker = markers[idx],label=cl)

    
from sklearn.svm import SVC



svm = SVC(kernel='rbf',random_state=0,gamma=.10,C=1)



svm.fit(x_train_std,y_train)



print ("The accuracy on training set is {}".format(svm.score(x_train_std,y_train)))

print ("The accuracy on test set is {}".format(svm.score(x_test_std,y_test)))
import warnings



def versiontuple(v):

    return tuple(map(int,(v.split('.'))))



def plot_decision_regions(x,y,classifier,test_idx=None,resolution=0.02):

    

    markers = ('s','x','o','^','v')

    colors = ["red", "blue","lightgreen","grey","cyan"]

    

    cmap = ListedColormap(colors[:len(np.unique(y))])

    

    x1_min,x1_max = x[:, 0].min()-1,x[:, 0].max()+1

    x2_min,x2_max = x[:, 1].min()-1,x[:, 1].max()+1

    

    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),

                          np.arange(x2_min,x2_max,resolution))

    

    z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)

    z = z.reshape(xx1.shape)

    

    plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)

    plt.xlim(xx1.min(),xx1.max())

    plt.ylim(xx2.min(),xx2.max())

    

    for idx,cl in enumerate(np.unique(y)):

        plt.scatter(x=x[y==cl,0],y=x[y==cl,1],

                   alpha = 0.8,c=cmap(idx),

                   marker=markers[idx],label=cl)

    
plot_decision_regions(x_test_std,y_test,svm)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')

knn.fit(x_train_std,y_train)



print ("The accuracy on training set is {}".format(knn.score(x_train_std,y_train)))

print ("The accuracy on test set is {}".format(knn.score(x_test_std,y_test)))
plot_decision_regions(x_test_std,y_test,knn)
import sklearn.xgboost as xgb



xgb_clf = xgb.XGBClassifier()

xgb_clf = xgb_clf.fit(x_train_std,y_train)



print ("The accuracy on training set is {}".format(xgb_clf.score(x_train_std,y_train)))

print ("The accuracy on test set is {}".format(xgb_clf.score(x_test_std,y_test)))
import pandas as pd

import numpy as np

from sklearn import datasets



%pylab inline



pylab.rcParams['figure.figsize'] = (10,6)



iris = datasets.load_iris()



x = iris.data[:,[2,3]]

y = iris.target



iris_df = pd.DataFrame(iris.data[:,[2,3]],columns = iris.feature_names[2:])



print (iris_df.head())



print ('\n' + "The uniques labels in the data are" + str(np.unique(y)))

from sklearn.cross_validation import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)



print ("There are {} trainig samples and {} test samples".format(x_train.shape[0],x_test.shape[0]))

print()
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

sc.fit(x_train)



x_train_std = sc.transform(x_train)

x_test_std = sc.transform(x_test)



print (pd.DataFrame(x_train_std, columns = iris_df.columns).head())
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt



markers = ('s','x','o')

colors = ('red','blue','green')



cmap = ListedColormap(colors[:len(np.unique(y_test))])



for idx, cl in enumerate(np.unique(y)):

    plt.scatter(x = x[y==cl,0],y=x[y==cl,1],

               c = cmap(idx),marker = markers[idx],label=cl)

    
from sklearn.svm import SVC



svm = SVC(kernel='rbf',random_state=0,gamma=.10,C=1)



svm.fit(x_train_std,y_train)



print ("The accuracy on training set is {}".format(svm.score(x_train_std,y_train)))

print ("The accuracy on test set is {}".format(svm.score(x_test_std,y_test)))
import warnings



def versiontuple(v):

    return tuple(map(int,(v.split('.'))))



def plot_decision_regions(x,y,classifier,test_idx=None,resolution=0.02):

    

    markers = ('s','x','o','^','v')

    colors = ["red", "blue","lightgreen","grey","cyan"]

    

    cmap = ListedColormap(colors[:len(np.unique(y))])

    

    x1_min,x1_max = x[:, 0].min()-1,x[:, 0].max()+1

    x2_min,x2_max = x[:, 1].min()-1,x[:, 1].max()+1

    

    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),

                          np.arange(x2_min,x2_max,resolution))

    

    z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)

    z = z.reshape(xx1.shape)

    

    plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)

    plt.xlim(xx1.min(),xx1.max())

    plt.ylim(xx2.min(),xx2.max())

    

    for idx,cl in enumerate(np.unique(y)):

        plt.scatter(x=x[y==cl,0],y=x[y==cl,1],

                   alpha = 0.8,c=cmap(idx),

                   marker=markers[idx],label=cl)

    
plot_decision_regions(x_test_std,y_test,svm)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')

knn.fit(x_train_std,y_train)



print ("The accuracy on training set is {}".format(knn.score(x_train_std,y_train)))

print ("The accuracy on test set is {}".format(knn.score(x_test_std,y_test)))
plot_decision_regions(x_test_std,y_test,knn)