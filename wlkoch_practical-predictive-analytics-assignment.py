#Load any modules that might be needed in the analysis



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sc

import sklearn as sk

import random





from scipy import stats

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score





import warnings

warnings.filterwarnings('ignore')

%matplotlib inline





# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#load training and test files and do some preliminary data exploration



# load training data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')





def rstr(df): return df.shape, df.apply(lambda x: [x.unique()])





print("\nExamine basic information about training file")

print("--------------------------------------------------\n")

df_train.info()





print("\n\nLook at data values - including categorical values")

print("--------------------------------------------------\n")

print(rstr(df_train))



print("\n\nGet a quick look at some basic stats for numerical data")

print("--------------------------------------------------\n")

df_train.describe()
from sklearn import tree

from sklearn.cross_validation import train_test_split





clean = df_train.dropna()

clean["Gender"] = (clean["Sex"] == "female")*1









data_1 = clean[["Pclass", "Age"]]

target_1 = clean[["Survived"]]



x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split( data_1, target_1, test_size = 0.3, random_state = 100)



clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,

                               max_depth=10, min_samples_leaf=15)



clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,

                                max_depth=3, min_samples_leaf=5)





clf_gini = clf_gini.fit(x_train_1, y_train_1)

y_pred_1 = clf_gini.predict(x_test_1)





clf_entropy = clf_entropy.fit(x_train_1, y_train_1)

y_pred_1_entropy = clf_entropy.predict(x_test_1)







print ("Model 1 (gini) Accuracy is ", accuracy_score(y_test_1,y_pred_1)*100)

print ("Confusion matrix for this model is: \n")

print (confusion_matrix(y_test_1, y_pred_1), "\n\n")



print ("Model 1 (entropy) Accuracy is ", accuracy_score(y_test_1,y_pred_1_entropy)*100)

print ("Confusion matrix for this model is: \n")

print (confusion_matrix(y_test_1, y_pred_1_entropy), "\n\n\n\n\n")

















data_2 = clean[["Pclass", "Age", "Gender"]]

target_2 = clean[["Survived"]]



x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split( data_2, target_2, test_size = 0.3, random_state = 100)



clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,

                               max_depth=10, min_samples_leaf=15)



clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,

                                max_depth=3, min_samples_leaf=5)





clf_gini = clf_gini.fit(x_train_2, y_train_2)

y_pred_2 = clf_gini.predict(x_test_2)



clf_entropy = clf_entropy.fit(x_train_2, y_train_2)

y_pred_2_entropy = clf_entropy.predict(x_test_2)





print ("Model 2 (gini) Accuracy is ", accuracy_score(y_test_2,y_pred_2)*100)

print ("Confusion matrix for this model is: \n")

print (confusion_matrix(y_test_2, y_pred_2), "\n\n")





print ("Model 2 (entropy) Accuracy is ", accuracy_score(y_test_2,y_pred_2_entropy)*100)

print ("Confusion matrix for this model is: \n")

print (confusion_matrix(y_test_2, y_pred_2_entropy), "\n\n\n")















data_3 = clean[["Pclass", "Age", "Gender", "Parch"]]

target_3 = clean[["Survived"]]



x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split( data_3, target_3, test_size = 0.3, random_state = 100)



clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,

                               max_depth=15, min_samples_leaf=20)

clf_gini = clf_gini.fit(x_train_3, y_train_3)



y_pred_3 = clf_gini.predict(x_test_3)



#print "Model 2 Accuracy is ", accuracy_score(y_test_3,y_pred_3)*100


