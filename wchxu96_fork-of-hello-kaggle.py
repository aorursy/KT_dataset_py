# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA

from sklearn.preprocessing import Imputer

from sklearn.model_selection import KFold

from sklearn import linear_model

from sklearn.metrics import make_scorer

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import svm

from sklearn.metrics import r2_score

from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import tflearn

import tensorflow as tf

import seaborn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

#train.head()

test = pd.read_csv('../input/test.csv')

y = train.SalePrice

data = pd.concat([train,test],ignore_index=True)

data = data.drop("SalePrice",1)

print (data.shape)

#print (train.shape)

#print (test.shape)

#train.head()

#y = train.iloc[:,-1][:,np.newaxis]

#y.head()

#X = train.iloc[:,:-1]

#print (X)

#y.head()

#print (X.shape)

#print (y.shape)

#X.isnull().sum()

nans = pd.isnull(data).sum()

nans = nans[nans>0]

print (nans)

#print (nans.columns)
'''from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train_reduced = PCA(n_components=2).fit_transform(X_train)

fig = plt.figure(1,figsize=(8,6))

ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(X_train_reduced[:,0],X_train_reduced[:,1],y_train[:],c='r',cmap=plt.cm.Paired)

ax.set_title("First two PCA directions")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("the price")

ax.w_zaxis.set_ticklabels([])

plt.show()

'''
data = data.drop('Id',1)

data = data.drop('Alley',1)

data = data.drop('Fence',1)

data = data.drop('FireplaceQu',1)

data = data.drop('MiscFeature',1)

data = data.drop('PoolQC',1)
data.shape
data.dtypes.value_counts()
all_columns = data.columns.values

non_categorical = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", 

                   "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", 

                   "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea", 

                   "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", 

                   "ScreenPorch","PoolArea", "MiscVal"]

categorical = [x for x in all_columns if x not in non_categorical]
data = pd.get_dummies(data)

imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)

data = imp.fit_transform(data)

# log transformation I do not understand

data = np.log(data)

labels = np.log(y)

data[data==-np.inf] = 0
pca = PCA(whiten=True)

pca.fit(data)

variance = pd.DataFrame(pca.explained_variance_ratio_)

print (np.cumsum(pca.explained_variance_ratio_))
pca = PCA(n_components=36,whiten=True)

pca = pca.fit(data)

pca_data = pca.transform(data)

pca_data.shape
pca_train = pca_data[:1460,:]

pca_test = pca_data[1460:,:]
def lets_try(train,labels):

    results={}

    def test_model(clf):

        

        cv = KFold(n_splits=5,shuffle=True,random_state=45)

        r2 = make_scorer(r2_score)

        r2_val_score = cross_val_score(clf, train, labels, cv=cv,scoring=r2)

        scores=[r2_val_score.mean()]

        return scores



    clf = linear_model.LinearRegression()

    results["Linear"]=test_model(clf)

    

    clf = linear_model.Ridge()

    results["Ridge"]=test_model(clf)

    

    clf = linear_model.BayesianRidge()

    results["Bayesian Ridge"]=test_model(clf)

    

    clf = linear_model.HuberRegressor()

    results["Hubber"]=test_model(clf)

    

    clf = linear_model.Lasso(alpha=1e-4)

    results["Lasso"]=test_model(clf)

    

    clf = BaggingRegressor()

    results["Bagging"]=test_model(clf)

    

    clf = RandomForestRegressor()

    results["RandomForest"]=test_model(clf)

    

    clf = AdaBoostRegressor()

    results["AdaBoost"]=test_model(clf)

    

    clf = svm.SVR()

    results["SVM RBF"]=test_model(clf)

    

    clf = svm.SVR(kernel="linear")

    results["SVM Linear"]=test_model(clf)

    

    results = pd.DataFrame.from_dict(results,orient='index')

    results.columns=["R Square Score"] 

    results=results.sort(columns=["R Square Score"],ascending=False)

    results.plot(kind="bar",title="Model Scores")

    axes = plt.gca()

    axes.set_ylim([0.5,1])

    return results



lets_try(pca_train,labels)