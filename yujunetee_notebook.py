import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler , LabelEncoder
from scipy import stats
import warnings
import keras
import random
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import (roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

'''#read train and test dataset
df = pd.read_csv("/content/drive/Shared drives/Data Science/train.csv")

#drop id column
df = df.drop(['ID'],axis=1)

X  = df.iloc[:,:-1] #drop the last column
y = df.iloc[:,-1] #choose the last column

Xtrain , Xtest , ytrain , ytest  = train_test_split(X,y,test_size=0.2, random_state=2)

print('There are {} samples in the training set and {} samples in the test set.'.format(
Xtrain.shape[0] , Xtest.shape[0]))
print()'''
#read train and test dataset
df1 = pd.read_csv("../input/datasetfolders/train.csv")

#drop id column in order to prevent data leakage
df1 = df1.drop(['ID'],axis=1)

df2 = pd.read_csv("../input/datasetfolders/bank.csv" , sep=';' , engine='python')
df2.rename(columns={'y': 'subscribed'} , inplace = True)

#concat 2 datsets
X = pd.concat([df1,df2], ignore_index=False)
X = X.drop_duplicates()
X.head()

no_sub = X[X['subscribed'] == 'no'].index

count = 0;
sampling = []
for x in no_sub:
  if(count % 2 == 0):
    sampling.append(x)
  count = count + 1

X.drop(sampling,inplace=True)

no_sub = X[X['subscribed'] == 'no'].index

count = 0;
sampling = []
for x in no_sub:
  if(count % 2 == 0):
    sampling.append(x)
  count = count + 1

X.drop(sampling,inplace=True)

print(X['subscribed'].value_counts())

train , test = train_test_split(X,test_size=0.2, random_state=6)

#train dataset
Xtrain = train.iloc[:,:-1] #drop the last column
ytrain = train.iloc[:,-1] #choose the last column

#test dataset
Xtest = test.iloc[:,:-1] #drop the last column
ytest = test.iloc[:,-1] #choose the last column

print('There are {} samples in the training set and {} samples in the test set.'.format(
Xtrain.shape[0] , Xtest.shape[0]))
print()
Xtrain.head()
#to identify numerical and categorical data
Xtrain.info()
plt.figure(figsize=(20,10))
sns.heatmap(Xtrain.corr(),annot = True)
# desciptive analysis for numerical columns
Xtrain.describe()
#visualize the data distribution of numerical data

Xtrain[['age','balance','day','duration','campaign','pdays','previous']].hist(bins=15, figsize=(15, 6), layout=(2, 4));
def bivariate_distribution(title):
  train = pd.concat([Xtrain,ytrain],axis=1)

  sns.FacetGrid(train,hue='subscribed' ,size=5 ).map(sns.distplot,title).add_legend()


bivariate_distribution('age')
bivariate_distribution('balance')
bivariate_distribution('campaign')
bivariate_distribution('day')
bivariate_distribution('duration')
bivariate_distribution('pdays')
bivariate_distribution('previous')
def explore_categorical_column(title):
  train = pd.concat([Xtrain,ytrain],axis=1)

  sns.catplot(x=title,kind='count', hue="subscribed", palette='pastel', data=train)
explore_categorical_column("job")

job_train = pd.crosstab(Xtrain.job,ytrain)
print(job_train)
explore_categorical_column("marital")

marital_train = pd.crosstab(Xtrain.marital,ytrain)
print(marital_train)
explore_categorical_column("default")

default_train = pd.crosstab(Xtrain.default,ytrain)
print(default_train)
explore_categorical_column("housing")

housing_train = pd.crosstab(Xtrain.housing,ytrain)
print(housing_train)
explore_categorical_column("loan")

loan_train = pd.crosstab(Xtrain.loan,ytrain)
print(loan_train)
explore_categorical_column("contact")

contact_train = pd.crosstab(Xtrain.contact,ytrain)
print(contact_train)
explore_categorical_column("month")

month_train = pd.crosstab(Xtrain.month,ytrain)
print(month_train)
explore_categorical_column("poutcome")

poutcome_train = pd.crosstab(Xtrain.poutcome,ytrain)
print(poutcome_train)
job_train = pd.crosstab(Xtrain.job,ytrain)
marital_train = pd.crosstab(Xtrain.marital,ytrain)
edu_train = pd.crosstab(Xtrain.education,ytrain)
default_train = pd.crosstab(Xtrain.default,ytrain)
house_train = pd.crosstab(Xtrain.housing,ytrain)
loan_train = pd.crosstab(Xtrain.loan,ytrain)
contact_train = pd.crosstab(Xtrain.contact,ytrain)
month_train = pd.crosstab(Xtrain.month,ytrain)
poutcome_train = pd.crosstab(Xtrain.poutcome,ytrain)

#returns four values, 𝜒2 value, p-value, degree of freedom and expected values.

a = [job_train,marital_train,edu_train,default_train,house_train,loan_train,contact_train,month_train,poutcome_train]

print("P values of every column")
n=1
for x in a:
  
  chi, pval, dof, exp = chi2_contingency(x)
  significance = 0.05
  print(n,'. -------------------------------',x.index.name,'---------------------------------')

  print('p-value=%.6f, significance=%.2f\n' % (pval, significance))
  if pval < significance:
    print("""At %.2f level of significance, we reject the null hypotheses and accept H1. 
  They are not independent.""" % (significance))
  else:
    print("""At %.2f level of significance, we accept the null hypotheses. 
  They are independent.""" % (significance))
    
  #print(x.index.name," = " ,chi2_contingency(x)[1]) # print p values

  print('  --------------------------------------------------------------------------------\n\n')
  n+=1
def column_encoding(df_x , df_y):

  df = pd.concat([df_x,df_y],axis=1)
  
  label_encoder = preprocessing.LabelEncoder()

  nominal_cols = ['job', 'marital','education' , 'contact', 'poutcome']
  for name in nominal_cols:
    df[name] = label_encoder.fit_transform(df[name])
    df[name].value_counts()

  #encoding 'default' , 'housing', 'loan' attributes 
  # 1 is yes , 0 is no
  mapping_dictionary = {"default" :{"yes" : 1 , "no" : 0},
                      "housing"  :{"yes" : 1 , "no" : 0},
                      "loan" :{"yes" : 1 , "no" : 0} ,
                      "subscribed" : {"yes" : 1 , "no" : 0}}

  df = df.replace(mapping_dictionary)

  #month
  replace_dictionary = { "month" : {"jan" : 1 , 
                                  "feb" : 2,
                                  "mar" : 3,
                                  "apr" : 4,
                                  "may" : 5,
                                  "jun" : 6,
                                  "jul" : 7,
                                  "aug" : 8,
                                  "sep" : 9,
                                  "oct" : 10,
                                  "nov" : 11,
                                  "dec" : 12}}

  df.replace(replace_dictionary , inplace=True)

  df_y = df.subscribed
  df_x = df.drop('subscribed', axis=1)
  return df_x , df_y

Xtrain , ytrain = column_encoding(Xtrain , ytrain)
Xtest , ytest = column_encoding(Xtest , ytest)
Xtrain.head()
#handle missing value
null_counts = Xtrain.isnull().sum()
print("Number of null values in each column:\n{}".format(null_counts))

#conclusion : no missing value
# Detect duplicate data
Xtrain_dedupped = Xtrain.drop_duplicates()

print(Xtrain.shape)
print(Xtrain_dedupped.shape)
#remove outliers

from scipy import stats

X = pd.concat([Xtrain,ytrain],axis=1)

X = X[(np.abs(stats.zscore(X)) < 3).all(axis=1)]

Xtrain = X.iloc[:,:-1] #drop the last column
ytrain = X.iloc[:,-1] #choose the last column
print(X)
print(X.shape)
categorical_var_train = Xtrain[['contact','education','default','housing','loan','job','poutcome','marital',
                         'month']]

numerical_var_train = Xtrain.drop(['contact','education','default','housing','loan','job','poutcome','marital',
                         'month'],axis=1)    

categorical_var_test = Xtest[['contact','education','default','housing','loan','job','poutcome','marital',
                         'month']]

numerical_var_test = Xtest.drop(['contact','education','default','housing','loan','job','poutcome','marital',
                         'month'],axis=1)    
        
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
scaled_Xtrain = scaler.fit_transform(numerical_var_train)
scaled_Xtest = scaler.transform(numerical_var_test)
scaled_Xtrain = pd.DataFrame(scaled_Xtrain)
scaled_Xtrain = scaled_Xtrain.reset_index() 
categorical_var_train =  categorical_var_train.reset_index() 

scaled_Xtest = pd.DataFrame(scaled_Xtest)
scaled_Xtest = scaled_Xtest.reset_index() 
categorical_var_test = categorical_var_test.reset_index() 
Xtrain = pd.concat([scaled_Xtrain,categorical_var_train],axis=1)
Xtrain = Xtrain.drop(['index','index'],axis=1)   

Xtest = pd.concat([scaled_Xtest,categorical_var_test],axis=1)
Xtest = Xtest.drop(['index','index'],axis=1)   

Xtrain.head(-20)

from sklearn.naive_bayes import GaussianNB       # 1. choose model class
naive_model = GaussianNB()                       # 2. instantiate model
 
naive_model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_naive_model_model = naive_model.predict(Xtest) 
from sklearn.metrics import accuracy_score
#accuracy_score(ytest, y_naive_model_model)
print(f"Accuracy : {accuracy_score(ytest, y_naive_model_model)*100} %" )
#plt.scatter(Xtrain.iloc[:, 0], Xtrain.iloc[:, 1], c=ytrain, s=50, cmap='RdBu')
#lim = plt.axis()
#plt.scatter(Xtest.iloc[:, 0], Xtest.iloc[:, 1], c=ytest, s=20, cmap='RdBu', alpha=0.1)
#plt.axis(lim);
from sklearn.neighbors import KNeighborsClassifier

def grid_search_knn():

  #Grid Search to find the best parameters
  k_range = list(range(1,31))
  weight_options = ["uniform", "distance"]
  param_grid = dict(n_neighbors = k_range, weights = weight_options)

  knn = KNeighborsClassifier()
  grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')
  grid.fit(Xtrain,ytrain)

  #print(grid.grid_scores_)
  '''
  print(grid.grid_scores_[0].parameters)
  print(grid.grid_scores_[0].cv_validation_scores)
  print(grid.grid_scores_[0].mean_validation_score)
  '''

  print (grid.best_score_)
  print (grid.best_params_)
  print (grid.best_estimator_)

grid_search_knn()

knn = KNeighborsClassifier(n_neighbors = 14)

knn.fit(Xtrain, ytrain)

knn_model = knn.predict(Xtest)  
#knn.score(scaled_Xtest, ytest)
#accuracy_score(ytest, knn_model)
print(f"Accuracy : {accuracy_score(ytest, knn_model)*100} %" )
from sklearn.model_selection import RandomizedSearchCV

def randomize_search_decision_tree():

  X_train , X_test , y_train , y_test = Xtrain , Xtest , ytrain , ytest

  max_depth = list(range(1, 50))
  min_samples_leaf = list(range(1, 60))
  min_samples_split = list(range(2,50))
  max_features = list(range(1, X_train.shape[1]))
  criterion = ['entropy' , 'gini']


  decision_tree_model = DecisionTreeClassifier()

  #carry out randomized search
  parameter_grid = dict(criterion=criterion,
                      max_features=max_features,
                      min_samples_leaf=min_samples_leaf,
                      max_depth=max_depth,
                      min_samples_split=min_samples_split)

  grid = RandomizedSearchCV(estimator=decision_tree_model, param_distributions=parameter_grid)
  grid.fit(X_train,y_train)

  print("Best criterion ：" , grid.best_estimator_.criterion)
  print("Best max_features" , grid.best_estimator_.max_features)
  print("Best min_samples_leaf : " , grid.best_estimator_.min_samples_leaf)
  print("Best max_depth : " , grid.best_estimator_.max_depth )
  print("Best min_samples_split : " , grid.best_estimator_.min_samples_split)
criterion = "gini"
max_features = 10
min_samples_leaf = 41
max_depth = 35
min_samples_split = 41
#result of randomized search

X_train , X_test , y_train , y_test = Xtrain , Xtest , ytrain , ytest

decision_tree_model = DecisionTreeClassifier(criterion=criterion,
                      max_features=max_features,
                      min_samples_leaf=min_samples_leaf,
                      max_depth=max_depth,
                      min_samples_split=min_samples_split,
                      random_state = 50
                      )
decision_tree_model.fit(X_train, y_train)    

# Predicton on test dataset
y_pred_decision_tree = decision_tree_model.predict(X_test) 

print(f"Accuracy : {decision_tree_model.score(X_test,y_test)*100} %" )
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

def randomize_search_random_forest():

  X_train , X_test , y_train , y_test = Xtrain , Xtest , ytrain , ytest

  n_estimators = [100,200,300,400,500,600,700,800,900]
  max_features = ['auto', 'sqrt']
  max_depth = [10,20,30,40,50,60,70,80,90,100]
  min_samples_split = [10,20,30,40,50]
  min_samples_leaf = [10,20,30,40,50]
  bootstrap = [True, False]
  max_leaf_nodes = [2,4,6,8,10,20,30,40,50]

  random_forest_model = RandomForestClassifier()

  #carry out randomized search
  parameter_grid = dict(n_estimators=n_estimators,
                      max_features=max_features,
                      min_samples_leaf=min_samples_leaf,
                      max_depth=max_depth,
                      min_samples_split=min_samples_split,
                      bootstrap=bootstrap,
                      max_leaf_nodes=max_leaf_nodes)

  grid = RandomizedSearchCV(estimator=random_forest_model, param_distributions=parameter_grid)
  grid.fit(X_train,y_train)

  #print hyperparatmeter values
  print("Best n_estimators : " , grid.best_estimator_.n_estimators)
  print("Best max_features : " , grid.best_estimator_.max_features)
  print("Best min_samples_leaf : " , grid.best_estimator_.min_samples_leaf)
  print("Best max_depth : " , grid.best_estimator_.max_depth)
  print("Best min_samples_split : " , grid.best_estimator_.min_samples_split)
  print("Best bootstrap : " , grid.best_estimator_.bootstrap)
  print("Best max_leaf_nodes : " , grid.best_estimator_.max_leaf_nodes)

randomize_search_random_forest()
n_estimators = 500
max_features = "auto"
min_samples_leaf = 30
max_depth = 30
min_samples_split = 40
bootstrap = "True"
max_leaf_nodes = 50
#result of randomized search
random_forest_model = RandomForestClassifier(n_estimators=n_estimators ,
                      max_features=max_features ,
                      min_samples_leaf=min_samples_leaf,
                      max_depth=max_depth,
                      min_samples_split=min_samples_split ,
                      bootstrap=bootstrap,
                      max_leaf_nodes=max_leaf_nodes,
                      random_state = 50)
random_forest_model.fit(X_train, y_train)   

# Predicton on test with giniIndex 
y_pred_random_forest = random_forest_model.predict(X_test) 

print(f"Accuracy : {random_forest_model.score(X_test,y_test)*100} %" )

# Instantiate classifier
logistic_regression = LogisticRegression(random_state = 30)

# Set up hyperparameter grid for tuning
logistic_regression_param_grid = {'C' : [0.0001, 0.001, 0.01, 0.05, 0.1] }

# Tune hyperparameters
logistic_regression_model = GridSearchCV(logistic_regression, param_grid = logistic_regression_param_grid, cv = 5)

# Fit model to training data
logistic_regression_model.fit(Xtrain, ytrain)
# Predict test data on logistic regression
print(f"Accuracy : {logistic_regression_model.score(Xtest, ytest)*100} %" )

# Obtain model performance metrics
lr_pred_prob = logistic_regression_model.predict_proba(Xtest)[:,1]
lr_auroc = roc_auc_score(ytest, lr_pred_prob) 
# Create an object of sequential model
ann_classifier = Sequential()
# Add the first hidden layer
ann_classifier.add(Dense(9, activation = 'relu', input_dim = 16))
# Adding the second hidden layer
ann_classifier.add(Dense(9, activation= 'relu'))
# Adding the output layer
ann_classifier.add(Dense(1, activation = 'sigmoid'))
ann_classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Model is trained over 100 epochs
ann_model = ann_classifier.fit(Xtrain, ytrain, validation_split = 0.33, batch_size = 10, epochs= 100)
# Predict probabilities for test set
ann_y_predict = ann_classifier.predict(Xtest)
# Predict crisp classes for test set
ann_y_classes = ann_classifier.predict_classes(Xtest)
# Reduce to 1d array
ann_y_classes = ann_y_classes[:, 0]
# Model score calculation
ann_score = accuracy_score(ann_y_predict.astype('int'), ytest.astype('int'))
print(f"Accuracy : {ann_score*100} %" )
# Import necessary libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier

# Scalling the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(Xtrain)
X_test = scaler.transform(Xtest)

# Set different learning rates to retrieve the best rate on performance
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(Xtrain, ytrain)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(Xtrain, ytrain)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(Xtest, ytest)))
# The best learning rate was 0.5 to fit into the classifier
gradient_boosting_model = GradientBoostingClassifier(n_estimators=20, learning_rate=1, max_features=2, max_depth=2, random_state=0)
gradient_boosting_model.fit(Xtrain, ytrain)
y_pred_gradient_boosting = gradient_boosting_model.predict(Xtest)
gbScore = gradient_boosting_model.score(Xtest, ytest)
gbMatrix = confusion_matrix(ytest, y_pred_gradient_boosting)

# Output of accuracy
print(f"Accuracy : {gbScore*100} %")
# Import necessary library
from sklearn.svm import SVC

# Create a linear SVM classifier
svm_model = SVC(kernel='linear', probability=True)

# Train classfier
svm_model.fit(Xtrain, ytrain)

# Take the model that was trained on the Xtrain data and apply it to the Xtest
y_pred_svm = svm_model.predict(Xtest)

# Calculation of accuracy
svmScore = svm_model.score(Xtest, ytest)

# Calculation of confusion matrix
svmMatrix = confusion_matrix(ytest, y_pred_svm)

# Print output
print(f"Accuracy : {svmScore*100} %")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(naive_model, title, Xtrain, ytrain,ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()
title = "Learning Curves (KNN)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

estimator = KNeighborsClassifier()
plot_learning_curve(knn, title, Xtrain, ytrain,ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)



plt.show()
title = "Learning Curves (Decision Tree)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plot_learning_curve(decision_tree_model, title, Xtrain, ytrain,  ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()
title = "Learning Curves (Random Forest)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plot_learning_curve(random_forest_model, title, Xtrain, ytrain,  ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()
title = "Learning Curves (Logistic Regression)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

estimator = LogisticRegression(random_state = 30)
plot_learning_curve(logistic_regression_model, title, Xtrain, ytrain, ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()
#Accuracy and Loss Curves of ANN Model 
#Accuracy vs Value Accuracy
ann_model.history.keys()
# summarize history for accuracy
plt.plot(ann_model.history['accuracy'])
plt.plot(ann_model.history['val_accuracy'])
plt.title('ANN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('EPOCH')
plt.legend(['Train', 'Test'], loc = 'lower right')
plt.show()

#loss vs value loss
plt.plot(ann_model.history['loss'])
plt.plot(ann_model.history['val_loss'])
plt.title('ANN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
title = "Learning Curves (Gradient Boosting)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

estimator = GradientBoostingClassifier()
plot_learning_curve(gradient_boosting_model, title, Xtrain, ytrain,  ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()
title = "Learning Curves (Linear SVM)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

estimator = SVC()
plot_learning_curve(svm_model, title, Xtrain, ytrain, ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()
def model_evaluation(model,name):

  confusion_matrix = pd.crosstab(ytest, model, rownames=['Actual'], colnames=['Predicted'], margins = True)
  sns.heatmap(confusion_matrix, square=True, annot=True, fmt='d', cbar=False)
  plt.xlabel('Prediction label')
  plt.ylabel('True Label');
  plt.title(name)
  plt.yticks([0.5,1.5], [ 'NO', 'YES'],va='center')
  plt.xticks([0.5,1.5], [ 'NO', 'YES'],va='center')
  plt.show()

  target_names = ['No' , 'Yes']
  
  print ('Precision:', precision_score(ytest, model,pos_label=1))
  print ('Accuracy:', accuracy_score(ytest, model))
  print ('F1 score:', f1_score(ytest, model,pos_label=1))
  print ('Recall:', recall_score(ytest, model,pos_label=1))
  print ('\n clasification report:\n', classification_report(ytest,model,target_names=target_names))
model_evaluation(knn_model,"KNN")
model_evaluation(y_naive_model_model,"Naive Bayes")
model_evaluation(y_pred_decision_tree,"Decision Tree")
model_evaluation(y_pred_random_forest,"Random Forest")
lr_y_pred = logistic_regression_model.predict(Xtest)
model_evaluation(lr_y_pred,"Logistic Regression")
model_evaluation(ann_y_classes,"ANN")
model_evaluation(y_pred_gradient_boosting,"Gradient Boosting")
model_evaluation(y_pred_svm,"Linear SVM")
from sklearn.metrics import average_precision_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
# plot no skill and model precision-recall curves

 
# no skill model, stratified random class predictions
model = DummyClassifier(strategy='stratified')
model.fit(Xtrain, ytrain)
yhat = model.predict_proba(ytest)
naive_probs = yhat[:, 1]
# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(ytest, naive_probs)
auc_score = auc(recall, precision)
print('No Skill PR AUC: %.3f' % auc_score)

# decision tree model
dt_yhat = decision_tree_model.predict_proba(Xtest)
dt_model_probs = dt_yhat[:, 1]
# calculate the precision-recall auc
dt_precision, dt_recall, _ = precision_recall_curve(ytest, dt_model_probs)
dt_auc_score = auc(dt_recall, dt_precision)
print('Decision Tree PR AUC: %.3f' % dt_auc_score)

#================================================================================

#random forest model
rf_yhat = random_forest_model.predict_proba(Xtest)
rf_model_probs = rf_yhat[:, 1]
# calculate the precision-recall auc
rf_precision, rf_recall, _ = precision_recall_curve(ytest, rf_model_probs)
rf_auc_score = auc(rf_recall, rf_precision)
print('Random Forest PR AUC: %.3f' % rf_auc_score)

#================================================================================

#knn
knn_yhat = knn.predict_proba(Xtest)
knn_model_probs = knn_yhat[:, 1]
# calculate the precision-recall auc
knn_precision, knn_recall, _ = precision_recall_curve(ytest, knn_model_probs)
knn_auc_score = auc(knn_recall, knn_precision)
print('KNN PR AUC: %.3f' % knn_auc_score)

#================================================================================

#Naive Bayess
n_yhat = naive_model.predict_proba(Xtest)
n_model_probs = n_yhat[:, 1]
# calculate the precision-recall auc
n_precision, n_recall, _ = precision_recall_curve(ytest, n_model_probs)
n_auc_score = auc(n_recall, n_precision)
print('Naive Bayes PR AUC: %.3f' % n_auc_score)

#================================================================================

#Logistic Regression
lr_yhat = logistic_regression_model.predict_proba(Xtest)
lr_model_probs = lr_yhat[:,1]
# calculate the precision-recall auc
lr_precision, lr_recall, _ = precision_recall_curve(ytest, lr_model_probs)
lr_auc_score = auc(lr_recall, lr_precision)
print('Logistic Regression PR AUC: %.3f' % lr_auc_score)

#================================================================================

#ANN
ann_yhat = ann_classifier.predict_proba(Xtest)
ann_model_probs = ann_yhat[:,0]
# calculate the precision-recall auc
ann_precision, ann_recall, _ = precision_recall_curve(ytest, ann_model_probs)
ann_auc_score = auc(ann_recall, ann_precision)
print('ANN PR AUC: %.3f' % ann_auc_score)

#================================================================================

# Gradient Boosting Classification
gb_yhat = gradient_boosting_model.predict_proba(Xtest)
gb_model_probs = n_yhat[:, 1]
# calculate the precision-recall auc
n_precision, n_recall, _ = precision_recall_curve(ytest, gb_model_probs)
n_auc_score = auc(n_recall, n_precision)
print('Gradient Boosting PR AUC: %.3f' % n_auc_score)

#================================================================================

# Linear SVM
svm_yhat = svm_model.predict_proba(Xtest)
svm_model_probs = n_yhat[:, 1]
# calculate the precision-recall auc
n_precision, n_recall, _ = precision_recall_curve(ytest, svm_model_probs)
n_auc_score = auc(n_recall, n_precision)
print('Linear SVM PR AUC: %.3f' % n_auc_score)

#================================================================================



# calculate the no skill line as the proportion of the positive class
no_skill = len(ytest[ytest==1]) / len(ytest)
# plot the no skill precision-recall curve
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	
# plot decision tree model precision-recall curve
dt_precision, dt_recall, _ = precision_recall_curve(ytest, dt_model_probs)
pyplot.plot(dt_recall, dt_precision, marker='x', label='Decision tree')
 
#plot random forest model precision-recall curve
rf_precision, rf_recall, _ = precision_recall_curve(ytest, rf_model_probs)
pyplot.plot(rf_recall, rf_precision, marker='x', label='Random Forest')

#================================================================================

#plot knn model precision-recall curve
knn_precision, knn_recall, _ = precision_recall_curve(ytest, knn_model_probs)
pyplot.plot(knn_recall, knn_precision, marker='x', label='KNN')

#plot random forest model precision-recall curve
n_precision,n_recall, _ = precision_recall_curve(ytest, n_model_probs)
pyplot.plot(n_recall, n_precision, marker='x', label='Naive Bayes')

#================================================================================

#plot logistic regression model precision-recall curve
lr_precision, lr_recall, _ = precision_recall_curve(ytest, lr_model_probs)
pyplot.plot(lr_recall, lr_precision, marker='x', label='Logistic Regression')

#plot ann model precision-recall curve
ann_precision,ann_recall, _ = precision_recall_curve(ytest, ann_model_probs)
pyplot.plot(ann_recall, ann_precision, marker='x', label='ANN')

#================================================================================

#plot Gradient Boosting model precision-recall curve
gb_precision, gb_recall, _ = precision_recall_curve(ytest, gb_model_probs)
pyplot.plot(gb_recall, gb_precision, marker='x', label='Gradient Boosting')

#plot Linear SVM model precision-recall curve
svm_precision,svm_recall, _ = precision_recall_curve(ytest, svm_model_probs)
pyplot.plot(svm_recall, svm_precision, marker='x', label='Linear SVM')

#================================================================================

# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()