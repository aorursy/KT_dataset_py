%matplotlib inline

import numpy as np #import numpy for operating on matricies

import matplotlib.pyplot as plt # our standard matplotlib import 

from sklearn.datasets import load_boston # loads the boston dataset

from sklearn.model_selection import train_test_split # for test/train/split operations

from sklearn.feature_selection import * # import all feature selection functions

from sklearn.preprocessing import StandardScaler # import function to scale features using z-scores

from sklearn import linear_model #module for linear models

from sklearn.model_selection import * # tools for model selection

from sklearn import svm # support vector machine

from sklearn import ensemble # ensemble module -- can be used to call random forests

from sklearn import metrics # accuracy/performance metrics


boston = load_boston()

print(boston.data.shape)

print(boston.feature_names)

print(np.max(boston.target), np.min(boston.target), np.mean(boston.target))

print(boston.DESCR)
# dome more descriptives

print(boston.data[0])

print(np.max(boston.data), np.min(boston.data), np.mean(boston.data)) # this tells us about the max,min,mean values of the dataset as a whole
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25, random_state=33)
print(np.max(X_train), np.min(X_train), np.mean(X_train), np.max(y_train), np.min(y_train), np.mean(y_train))
fs=SelectKBest(score_func=f_regression,k=5)

X_new=fs.fit_transform(X_train,y_train)

print(dict(zip(boston.feature_names,fs.get_support())))



x_min, x_max = X_new[:,0].min() - .5, X_new[:, 0].max() + .5

y_min, y_max = y_train.min() - .5, y_train.max() + .5

#fig=plt.figure()

#fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)



# Two subplots, unpack the axes array immediately

fig, axes = plt.subplots(1,5) # a grid of plots arranged into 1 row and 5 columns. 

# We access the different plots through items in the list `axes`

fig.set_size_inches(12,12) # set  size of figure



for i in range(5):

    axes[i].set_aspect('equal')

    axes[i].set_title('Feature ' + str(i))

    axes[i].set_xlabel('Feature')

    axes[i].set_ylabel('Median house value')

    axes[i].set_xlim(x_min, x_max)

    axes[i].set_ylim(y_min, y_max)

    plt.sca(axes[i])

    plt.scatter(X_new[:,i],y_train)

X_train = StandardScaler().fit_transform(X_train) # fit_transform is a convenience function

# that computes the fit of the model 

# (i.e. the parameters of the scaling equation and then applies the transformation to the input data



#scalerX = StandardScaler().fit(X_train)

#scalery = StandardScaler().fit(y_train)



#X_train = scalerX.transform(X_train)

#y_train = StandardScaler().fit_transform((y_train)



X_test = StandardScaler().fit_transform(X_test)

#y_test = StandardScaler().fit_transform(y_test)



print(np.max(X_train), np.min(X_train), np.mean(X_train), np.max(y_train), np.min(y_train), np.mean(y_train))



def train_and_evaluate(mod, X_train, y_train):

    

    mod.fit(X_train, y_train)

    yhat=mod.predict(X_train)

    MAE_scorer=metrics.make_scorer(metrics.mean_squared_error) # make a scorer to generate mean_squared_errors. 

    # We need to do this primarily to use the cross_val_score function

    print("model score on training data:",mod.score(X_train,y_train)) # this will provide a default model score that varies by the model/estimation technique used. 

    # OLS regression will score in terms of R^2

    print("training mean squred error:",MAE_scorer(mod,X_train,y_train))

    

    # create a k-fold croos validation iterator of k=5 folds

    scores = cross_val_score(mod, X_train, y_train, cv=5,scoring=MAE_scorer)

    print("average score 5-fold crossvalidation:",np.mean(scores))


mod_sgd = linear_model.SGDRegressor(loss='squared_loss', penalty=None,  random_state=42, shuffle=False)

print(mod_sgd)

train_and_evaluate(mod_sgd,X_train,y_train)

print(mod_sgd.coef_) #trailing underscore is sklearn code convention to indicate fitted/estimated values.

mod_sgd1 = linear_model.SGDRegressor(loss='squared_loss', penalty='l2',  random_state=42)

train_and_evaluate(mod_sgd1,X_train,y_train)
mod_sgd2 = linear_model.SGDRegressor(loss='squared_loss', penalty='l1',  random_state=42)

train_and_evaluate(mod_sgd2,X_train,y_train)
mod_sgd3 = linear_model.SGDRegressor(loss='squared_loss', penalty='elasticnet',  random_state=42)

train_and_evaluate(mod_sgd3,X_train,y_train)
mod_ridge = linear_model.Ridge()

train_and_evaluate(mod_ridge,X_train,y_train)


mod_svr= svm.SVR(kernel='linear') #essentially no transformation

train_and_evaluate(mod_svr,X_train,y_train)
mod_svr_poly= svm.SVR(kernel='poly') #polynomial basis expansion X+X^2+X^3+...

train_and_evaluate(mod_svr_poly,X_train,y_train)
mod_svr_rbf= svm.SVR(kernel='rbf') #radial basis function (e.g. gaussian kernel), 

# essentially indicates distance from a specific point

train_and_evaluate(mod_svr_rbf,X_train,y_train)
mod_svr_poly2= svm.SVR(kernel='poly',degree=2) # polynomial basis expansion limited to degree 2 X+X^2

train_and_evaluate(mod_svr_poly2,X_train,y_train)


mod_et=ensemble.ExtraTreesRegressor(n_estimators=10,random_state=42)

train_and_evaluate(mod_et,X_train,y_train)
print(mod_et.feature_importances_,boston.feature_names)


def measure_performance(X,y,clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True, show_r2_score=False):

    y_pred=clf.predict(X)   

    if show_accuracy:

        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n")



    if show_classification_report:

        print("Classification report")

        print(metrics.classification_report(y,y_pred),"\n")

        

    if show_confusion_matrix:

        print("Confusion matrix")

        print(metrics.confusion_matrix(y,y_pred),"\n")

        

    if show_r2_score:

        print("Coefficient of determination:{0:.3f}".format(metrics.r2_score(y,y_pred)),"\n")



        

measure_performance(X_test,y_test,mod_et, show_accuracy=False, show_classification_report=False,show_confusion_matrix=False, show_r2_score=True)

measure_performance(X_test,y_test,mod_svr_rbf, show_accuracy=False, show_classification_report=False,show_confusion_matrix=False, show_r2_score=True)