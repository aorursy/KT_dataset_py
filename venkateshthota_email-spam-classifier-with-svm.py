# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import validation_curve

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns
# load the data

email_rec = pd.read_csv("../input/spambase/realspambase.data",  sep = ',', header= None )

print(email_rec.head())
# renaming the columns

email_rec.columns  = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", 

                      "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet", 

                      "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will", 

                      "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free", 

                      "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit", 

                      "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", 

                      "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs", 

                      "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85", 

                      "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",

                      "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re", 

                      "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(", 

                      "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_hash", "capital_run_length_average", 

                      "capital_run_length_longest", "capital_run_length_total", "spam"]

print(email_rec.head())
# look at dimensions of the df

print(email_rec.shape)
# ensure that data type are correct

email_rec.info()
# there are no missing values in the dataset 

email_rec.isnull().sum()
# look at fraction of spam emails 

# 39.4% spams

email_rec['spam'].describe()
email_rec.describe()
# splitting into X and y

X = email_rec.drop("spam", axis = 1)

y = email_rec.spam.values.astype(int)
# scaling the features

# note that the scale function standardises each column, i.e.

# x = x-mean(x)/std(x)



from sklearn.preprocessing import scale

X = scale(X)
# split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)
# confirm that splitting also has similar distribution of spam and ham 

# emails

print(y_train.mean())

print(y_test.mean())
# Model building



# instantiate an object of class SVC()

# note that we are using cost C=1

model = SVC(C = 1)



# fit

model.fit(X_train, y_train)



# predict

y_pred = model.predict(X_test)
# Evaluate the model using confusion matrix 

from sklearn import metrics

metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
# print other metrics



# accuracy

print("accuracy", metrics.accuracy_score(y_test, y_pred))



# precision

print("precision", metrics.precision_score(y_test, y_pred))



# recall/sensitivity

print("recall", metrics.recall_score(y_test, y_pred))

# specificity (% of hams correctly classified)

print("specificity", 811/(811+38))
# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 4)



# instantiating a model with cost=1

model = SVC(C = 1)
# computing the cross-validation scores 

# note that the argument cv takes the 'folds' object, and

# we have specified 'accuracy' as the metric



cv_results = cross_val_score(model, X_train, y_train, cv = folds, scoring = 'accuracy') 
# print 5 accuracies obtained from the 5 folds

print(cv_results)

print("mean accuracy = {}".format(cv_results.mean()))
# specify range of parameters (C) as a list

params = {"C": [0.1, 1, 10, 100, 1000]}



model = SVC()



# set up grid search scheme

# note that we are still using the 5 fold CV scheme we set up earlier

model_cv = GridSearchCV(estimator = model, param_grid = params, 

                        scoring= 'accuracy', 

                        cv = folds, 

                        verbose = 1,

                       return_train_score=True)      
# fit the model - it will fit 5 folds across all values of C

model_cv.fit(X_train, y_train)  
# results of grid search CV

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# plot of C versus train and test scores



plt.figure(figsize=(8, 6))

plt.plot(cv_results['param_C'], cv_results['mean_test_score'])

plt.plot(cv_results['param_C'], cv_results['mean_train_score'])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')
best_score = model_cv.best_score_

best_C = model_cv.best_params_['C']



print(" The highest test accuracy is {0} at C = {1}".format(best_score, best_C))
# model with the best value of C

model = SVC(C=best_C)



# fit

model.fit(X_train, y_train)



# predict

y_pred = model.predict(X_test)
# metrics

# print other metrics



# accuracy

print("accuracy", metrics.accuracy_score(y_test, y_pred))



# precision

print("precision", metrics.precision_score(y_test, y_pred))



# recall/sensitivity

print("recall", metrics.recall_score(y_test, y_pred))

# specify params

params = {"C": [0.1, 1, 10, 100, 1000]}



# specify scores/metrics in an iterable

scores = ['accuracy', 'precision', 'recall']



for score in scores:

    print("# Tuning hyper-parameters for {}".format(score))

    

    # set up GridSearch for score metric

    clf = GridSearchCV(SVC(), 

                       params, 

                       cv=folds,

                       scoring=score,

                       return_train_score=True)

    # fit

    clf.fit(X_train, y_train)



    print(" The highest {0} score is {1} at C = {2}".format(score, clf.best_score_, clf.best_params_))

    print("\n")
import pandas as pd

import numpy as np

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import scale
email_rec = pd.read_csv("../input/spambase/realspambase.data",  sep = ',', header= None )

email_rec.head()
# renaming the columns

email_rec.columns  = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", 

                      "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet", 

                      "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will", 

                      "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free", 

                      "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit", 

                      "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", 

                      "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs", 

                      "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85", 

                      "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",

                      "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re", 

                      "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(", 

                      "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_hash", "capital_run_length_average", 

                      "capital_run_length_longest", "capital_run_length_total", "spam"]

print(email_rec.head())
# splitting into X and y

X = email_rec.drop("spam", axis = 1)

y = email_rec.spam.values.astype(int)
# scaling the features

X_scaled = scale(X)



# train test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 4)
# using rbf kernel, C=1, default value of gamma



model = SVC(C = 1, kernel='rbf')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# confusion matrix

confusion_matrix(y_true=y_test, y_pred=y_pred)
# accuracy

print("accuracy", metrics.accuracy_score(y_test, y_pred))



# precision

print("precision", metrics.precision_score(y_test, y_pred))



# recall/sensitivity

print("recall", metrics.recall_score(y_test, y_pred))
# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 4)



# specify range of hyperparameters

# Set the parameters by cross-validation

hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]}]





# specify model

model = SVC(kernel="rbf")



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = model, 

                        param_grid = hyper_params, 

                        scoring= 'accuracy', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      



# fit the model

model_cv.fit(X_train, y_train)                  

# cv results

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# converting C to numeric type for plotting on x-axis

cv_results['param_C'] = cv_results['param_C'].astype('int')



# # plotting

plt.figure(figsize=(16,6))



# subplot 1/3

plt.subplot(131)

gamma_01 = cv_results[cv_results['param_gamma']==0.01]



plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])

plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.01")

plt.ylim([0.80, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 2/3

plt.subplot(132)

gamma_001 = cv_results[cv_results['param_gamma']==0.001]



plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])

plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.001")

plt.ylim([0.80, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')





# subplot 3/3

plt.subplot(133)

gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]



plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])

plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.0001")

plt.ylim([0.80, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')

# printing the optimal accuracy score and hyperparameters

best_score = model_cv.best_score_

best_hyperparams = model_cv.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
# specify optimal hyperparameters

best_params = {"C": 100, "gamma": 0.0001, "kernel":"rbf"}



# model

model = SVC(C=100, gamma=0.0001, kernel="rbf")



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



# metrics

print(metrics.confusion_matrix(y_test, y_pred), "\n")

print("accuracy", metrics.accuracy_score(y_test, y_pred))

print("precision", metrics.precision_score(y_test, y_pred))

print("sensitivity/recall", metrics.recall_score(y_test, y_pred))