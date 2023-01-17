import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import scikitlearn 

# a famous library for machine learning

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

import graphviz
data = pd.read_csv("../input/HR_comma_sep.csv")
print(data.shape)

left = data.loc[data['left']==1]

print(left.shape)

left_ratio = data.shape[0] 
# Preprocessing



def convert_column(data,target_col):

    """

    convert string-based categorical var into number encoding. 

    """

    data_ = data.copy()

    target = data_[target_col].unique()

    num_encode = {name: n for n, name in enumerate(target)}

    data_['target'] = data_[target_col].replace(num_encode)

    

    return (data_, target)
data2, dept = convert_column(data, "sales")

data2 = data2.rename(columns = {'target':'dept'})

data2.drop('sales', axis=1, inplace=True)

print(data2.head(1))
data3, salary = convert_column(data2, "salary")

data3 = data3.rename(columns = {'target':'salary_level'})

data3.drop('salary', axis=1, inplace=True)

print(data3.head(1))
mon_hours = []

for i in range(len(data3)):

    if data3.iloc[i, 3] < 138:

        mon_hours.append(0)

    elif data3.iloc[i, 9] > 230:

        mon_hours.append(2)

    else:

        mon_hours.append(1)



data3.drop('average_montly_hours', axis=1, inplace=True)

data3['mon_hours'] = mon_hours

print(data3.head(1))
x_var = data3.drop('left', axis=1)

print(x_var.head(1))



y_var = data3['left']

print(y_var.head(1))


X_train, X_test, y_train, y_test = train_test_split(x_var, y_var, test_size=0.4, random_state=0)

print("Number of observation in the training set is", X_train.shape[0])

print(X_train.head(3))

new_index = list(range(0, X_train.shape[0] + 1))



X_train = X_train.values

y_train = y_train.values

print(X_train)
# k-fold cross validation with k = 5

kf = StratifiedKFold(n_splits=3)



for training, validation in kf.split(X_train, y_train):

    print("%s %s" % (training, validation), len(training), len(validation))



for training, validation in kf.split(X_train, y_train):

    X_training, X_validation = X_train[training], X_train[validation]

    y_training, y_validation = y_train[training], y_train[validation]

print(X_training)
# fit tree model



clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=round(0.075*X_train.shape[0]))

# 

clf = clf.fit(X_training, y_training)

print("Training set score is",clf.score(X_training, y_training))

print("Validation set score is",clf.score(X_validation, y_validation))
# Predict with testing set

pred = clf.predict(X_test)

result = pd.DataFrame({'Predicting result': pred,'Actual result': y_test})

print(result)

print("Testing set score is", clf.score(X_test, y_test))
x_var_names = list(x_var.columns)
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=x_var_names,class_names=True,

                               filled=True, rounded=True)  

graph = graphviz.Source(dot_data)  

graph 

# Why is my graph so freaking huge???

# Because I have a gigantic tree

#TreePlanter

# The line above is simply a hashtag. :)