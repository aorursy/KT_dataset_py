import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



data = pd.read_csv('../input/data.csv')
data.columns
print(f'Dimension of our data {data.shape} \n Data feature informations')

data.info()
data.head()
data.drop('id',axis=1, inplace=True)

data.drop(data.columns[[-1]], axis = 1, inplace=True)



data.info()
print(f' Total data {data.shape[0]} diagnosis.\n Categories: {data.diagnosis.unique()}\n Data value count: \n{data.diagnosis.value_counts()}')
data.describe()
data.var()
data.skew()
data.kurtosis()
data.info()
data[[x for x in data.columns if 'mean' in x] + ['diagnosis']].groupby('diagnosis').plot(figsize=(18, 6), label=True)
data_mean = data.iloc[:,1:11]
data_mean.info()
data_mean.hist(bins=10, figsize=(18,10))


bins = 12

plt.figure(figsize=(15,15))

for i, feature in enumerate(data_mean):

    rows = int(len(data_mean)/2)

    

    plt.subplot(rows, 2, i+1)

    

    sns.distplot(data[data['diagnosis']=='M'][feature], bins=bins, color='red', label='M');

    sns.distplot(data[data['diagnosis']=='B'][feature], bins=bins, color='blue', label='B');

    

    plt.legend(loc='upper right')



plt.tight_layout()

plt.show()
data_mean.plot(kind='density', subplots=True, sharex=False, sharey=False,layout=(4,3),fontsize=12, figsize=(16,10))
data[[x for x in data.columns if 'mean' in x] + ['diagnosis']].groupby('diagnosis').plot(kind= 'box' , subplots=True, layout=(4,3), sharex=False, sharey=False,figsize=(15,10))
data_mean.corr()
plt.figure(figsize=(10,10))

sns.heatmap(data_mean.corr(), annot=True, square=True, cmap='coolwarm')

plt.title('Breast Cancer Feature Correlation')

plt.show()
sns.pairplot(data ,hue='diagnosis', vars = data_mean.columns)
features_selection = ['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean']
diag_map = {'M':1, 'B':0}

data['diagnosis'] = data['diagnosis'].map(diag_map)



#from sklearn.preprocessing import LabelEncoder

#LE = LabelEncoder()

#data['diagnosis'] = LE.fit_transform(data['diagnosis'])
X_feature = data.loc[:,features_selection]

y = data.loc[:,'diagnosis']
from sklearn.preprocessing import StandardScaler



# Normalize the  data (center around 0 and scale to remove the variance).

scaler =StandardScaler()

X = scaler.fit_transform(X_feature)
# Fisrt we will standardize our data

X_pca = data.iloc[:,1:]

X_pcas = scaler.fit_transform(X_pca)
from sklearn.decomposition import PCA



# feature extraction

pca = PCA(n_components=10)

fit = pca.fit(X_pcas)

X_pca = fit.transform(X_pcas)
fit.explained_variance_ratio_ * 10, fit.components_
X_pca = pca.transform(X_pcas)



PCA_df = pd.DataFrame()



PCA_df['PCA_1'] = X_pca[:,0]

PCA_df['PCA_2'] = X_pca[:,1]

# diag_map = {'M':1, 'B':0}

plt.plot(PCA_df['PCA_1'][data.diagnosis == 1],PCA_df['PCA_2'][data.diagnosis == 1],'o', alpha = 0.7, color = 'r')

plt.plot(PCA_df['PCA_1'][data.diagnosis == 0],PCA_df['PCA_2'][data.diagnosis == 0],'o', alpha = 0.7, color = 'b')



plt.xlabel('PCA_1')

plt.ylabel('PCA_2')

plt.legend(['Malignant','Benign'])

plt.show()
#The amount of variance that each PC explains

var= pca.explained_variance_ratio_

#Cumulative Variance explains

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(var,var1)
plt.figure(figsize=(10,6))

plt.plot(var)

plt.title('Scree Plot')

plt.xlabel('Principal Component')

plt.ylabel('Eigenvalue')

leg = plt.legend(['Eigenvalues from PCA'], loc='best', borderpad=0.3,shadow=False,markerscale=0.4)

plt.grid(True)

#leg.get_frame().set_alpha(0.4)

#leg.draggable(state=True)

plt.show()
plt.figure(figsize=(10,6))

plt.plot(var1)

plt.title('Scree Plot')

plt.xlabel('Principal Component')

plt.ylabel('Eigenvalue')

leg = plt.legend(['Eigenvalues from PCA'], loc='best', borderpad=0.3,shadow=False,markerscale=0.4)

plt.grid(True)

#leg.get_frame().set_alpha(0.4)

#leg.draggable(state=True)

plt.show()
# Difference in numbers of columns after applying PCA

print("Before          After")

print(X_pcas.shape,'--->', X_pca.shape)
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
# Standardized data

# X_pcas and y



#  Divide records in training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(X_pcas, y, test_size=0.3, random_state=2, stratify=y)



#  Create an SVM classifier and train it on 70% of the data set.

clf = SVC(probability=True, gamma='auto')

clf.fit(X_train, y_train)



# Analyze accuracy of predictions on 30% of the holdout test sample.

classifier_score = clf.score(X_test, y_test)

print('\nThe classifier accuracy score is {:03.2f}\n'.format(classifier_score))
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix

from sklearn import metrics, preprocessing

from sklearn.metrics import classification_report


# Get average of 3-fold cross-validation score using an SVC estimator.

n_folds = 3

cv_error = list(cross_val_score(SVC(gamma='auto'), X_pcas, y, cv=n_folds))

print('\nThe {}-fold cross-validation accuracy score for this classifier is {}'.format(n_folds, cv_error))

print('Average score: {:.2f} %'.format(np.average(cv_error)))
from sklearn.feature_selection import SelectKBest, f_regression

clf2 = make_pipeline(SelectKBest(f_regression, k=3),SVC(probability=True, gamma='auto'))



scores = cross_val_score(clf2, X_pcas, y, cv=3)

print(scores)

avg = (100*np.mean(scores), 100*np.std(scores))

print("Average score and uncertainty: (%.2f +- %.3f)%%"%avg)
# The confusion matrix helps visualize the performance of the algorithm.

y_pred = clf.fit(X_train, y_train).predict(X_test)

cm = metrics.confusion_matrix(y_test, y_pred)

pd.crosstab(y_test, y_pred)
print(classification_report(y_test, y_pred ))


from sklearn.metrics import roc_curve, auc

# Plot the receiver operating characteristic curve (ROC).

fig, ax = plt.subplots(1,1, figsize=(6,6))

probas_ = clf.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])



ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))

ax.legend([auc(fpr,tpr)])

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_title('Receiver operating characteristic example')
from sklearn.model_selection import GridSearchCV
# Here I'm using orinigianl data X

Xs = data.iloc[:,1:]

X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=147, stratify=y)



# Split-out validation dataset

array = data.values

X = array[:,1:31]

y = array[:,0]



# Divide records in training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

import time

# Train classifiers.

start = time.time()

kernel_values = [ 'linear' , 'poly' ]

param_grid = {'C': np.logspace(-3, 3, 1), 'gamma': np.logspace(-3, 3, 1),'kernel': kernel_values}

# We are using n_flods = 5

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5,verbose=3, n_jobs=1)

grid.fit(X_train, y_train)



print(time.time()-start)
# Train classifiers.

#start = time.time()

#kernel_values = [ 'linear' ,  'poly' ,  'rbf' ,  'sigmoid' ]

#param_grid = {'C': np.logspace(-3, 3, 3), 'gamma': np.logspace(-3, 3, 3),'kernel': kernel_values}

# We are using n_flods = 5

#grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5,verbose=3, n_jobs=1)

#grid.fit(X_train, y_train)

print(time.time()-start)
#print("The best parameters are %s with a score of %0.2f"% (grid.best_params_, grid.best_score_))

#clf = grid.best_estimator_
#y_pred = clf.fit(X_train, y_train).predict(X_test)



#print(classification_report(y_test, y_pred))
#pd.crosstab(y_test, y_pred)
Xtrain = X_train[:, :2] # we only take the first two features to visualize in 2D graph.



# We create an instance of SVM and fit out data. 

# We do not scale our data since we want to plot the support vectors





C = 1.0  # SVM regularization parameter



svm = SVC(kernel='linear', random_state=0, gamma=0.1, C=C).fit(Xtrain, y_train)

rbf_svc = SVC(kernel='rbf', gamma=0.7, C=C).fit(Xtrain, y_train)

poly_svc = SVC(kernel='poly', degree=3, C=C).fit(Xtrain, y_train)
 # create a mesh to plot in

x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1

y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),

                         np.arange(y_min, y_max, 0.1))



# title for the plots

titles = ['SVC with linear kernel',

          'SVC with RBF kernel',

          'SVC with polynomial (degree 3) kernel']
plt.rcParams['figure.figsize'] = (15, 9) 

for i, clf in enumerate((svm, rbf_svc, poly_svc)):

    # Plot the decision boundary. For that, we will assign a color to each

    # point in the mesh [x_min, x_max]x[y_min, y_max].

    plt.subplot(2, 2, i + 1)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)



    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])



    # Put the result into a color plot

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)



    # Plot also the training points

    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=y_train, cmap=plt.cm.copper)

    plt.xlabel('radius_mean')

    plt.ylabel('texture_mean')

    plt.xlim(xx.min(), xx.max())

    plt.ylim(yy.min(), yy.max())

    plt.xticks(())

    plt.yticks(())

    plt.title(titles[i])



plt.show()

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.model_selection import KFold
# Spot-Check Algorithms

models = []

models.append(( 'LR' , LogisticRegression(solver='liblinear')))

models.append(( 'KNN' , KNeighborsClassifier(n_neighbors=5)))

models.append(( 'CART' , DecisionTreeClassifier()))

models.append(( 'NB' , GaussianNB()))

models.append(( 'SVM' , SVC(gamma='auto')))



# Test options and evaluation metric

num_folds = 10



seed = 147 

scoring =  'accuracy'





results = []

names = []

for name, model in models:

 kfold = KFold( n_splits=num_folds, random_state=seed)

 cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

 results.append(cv_results)

 names.append(name)

 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

 print(msg)

print('-> 10-Fold cross-validation accurcay score for the training data for six classifiers')
# Compare Algorithms

fig = plt.figure()

fig.suptitle( 'Algorithm Comparison' )

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
from sklearn.pipeline import Pipeline
# Standardize the dataset

pipelines = []

pipelines.append(( 'ScaledLR' , Pipeline([( 'Scaler' , StandardScaler()),( 'LR' ,

    LogisticRegression())])))

pipelines.append(( 'ScaledKNN' , Pipeline([( 'Scaler' , StandardScaler()),( 'KNN' ,

    KNeighborsClassifier())])))

pipelines.append(( 'ScaledCART' , Pipeline([( 'Scaler' , StandardScaler()),( 'CART' ,

    DecisionTreeClassifier())])))

pipelines.append(( 'ScaledNB' , Pipeline([( 'Scaler' , StandardScaler()),( 'NB' ,

    GaussianNB())])))

pipelines.append(( 'ScaledSVM' , Pipeline([( 'Scaler' , StandardScaler()),( 'SVM' , SVC())])))



results = []

names = []

for name, model in pipelines:

  kfold = KFold( n_splits=num_folds, random_state=seed)

  cv_results = cross_val_score(model, X_train, y_train, cv=kfold,

      scoring=scoring)

  results.append(cv_results)

  names.append(name)

  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

  print(msg)
# Compare Algorithms

fig = plt.figure()

fig.suptitle( 'Scaled Algorithm Comparison' )

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
#Make Support Vector Classifier Pipeline

pipe_svc = Pipeline([('scl', StandardScaler()),

                     ('pca', PCA(n_components=2)),

                     ('clf', SVC(probability=True))])



#Fit Pipeline to training Data

pipe_svc.fit(X_train, y_train)



#print('--> Fitted Pipeline to training Data')



scores = cross_val_score(estimator=pipe_svc, X=X_train, y=y_train, cv=10, n_jobs=1, verbose=0)

print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))



#Tune Hyperparameters

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},

              {'clf__C': param_range,'clf__gamma': param_range,

               'clf__kernel': ['rbf']}]

gs_svc = GridSearchCV(estimator=pipe_svc,

                  param_grid=param_grid,

                  scoring='accuracy',

                  cv=10,

                  n_jobs=1,verbose=3)

gs_svc = gs_svc.fit(X_train, y_train)


print('--> Tuned Parameters Best Score: ',gs_svc.best_score_)

print('--> Best Parameters: \n',gs_svc.best_params_)
from sklearn.neighbors import KNeighborsClassifier as KNN



pipe_knn = Pipeline([('scl', StandardScaler()),

                     ('pca', PCA(n_components=2)),

                     ('clf', KNeighborsClassifier())])

            

#Fit Pipeline to training Data

pipe_knn.fit(X_train, y_train) 



scores = cross_val_score(estimator=pipe_knn, 

                         X=X_train, 

                         y=y_train, 

                         cv=10,

                         n_jobs=1)

print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))



#Tune Hyperparameters

param_range = range(1, 31)

param_grid = [{'clf__n_neighbors': param_range}]

# instantiate the grid

grid = GridSearchCV(estimator=pipe_knn, 

                    param_grid=param_grid, 

                    cv=10, 

                    scoring='accuracy')

gs_knn = grid.fit(X_train, y_train)


print('--> Tuned Parameters Best Score: ',gs_knn.best_score_)

print('--> Best Parameters: \n',gs_knn.best_params_)


pipe_lr = Pipeline([('scl', StandardScaler()),

                     ('pca', PCA(n_components=2)),

                     ('clf', LogisticRegression())])

            

#Fit Pipeline to training Data

pipe_lr.fit(X_train, y_train) 



scores = cross_val_score(estimator=pipe_lr, 

                         X=X_train, 

                         y=y_train, 

                         cv=10,

                         n_jobs=1)

print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))



#Tune Hyperparameters

param_grid={"clf__C":np.logspace(-3,3,7), "clf__penalty":["l1","l2"]}# l1 lasso l2 ridge



# instantiate the grid

grid = GridSearchCV(estimator=pipe_lr, 

                    param_grid=param_grid, 

                    cv=10, 

                    scoring='accuracy')

gs_lr = grid.fit(X_train, y_train)


print('--> Tuned Parameters Best Score: ',gs_lr.best_score_)

print('--> Best Parameters: \n',gs_lr.best_params_)
#Use best parameters

clf_svc = gs_svc.best_estimator_



#Get Final Scores

clf_svc.fit(X_train, y_train)

scores = cross_val_score(estimator=clf_svc,

                         X=X_train,

                         y=y_train,

                         cv=10,

                         n_jobs=1)

print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))



print('--> Final Accuracy on Test set: %.5f' % clf_svc.score(X_test,y_test))
from sklearn.metrics import accuracy_score



clf_svc.fit(X_train, y_train)

y_pred = clf_svc.predict(X_test)



print("Accuracy:",accuracy_score(y_test, y_pred))



print(classification_report(y_test, y_pred))
pd.crosstab(y_test, y_pred)