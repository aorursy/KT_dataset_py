%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os



print(os.listdir("../input"))
df = pd.read_csv('../input/Iris.csv')

df.head(5) # Looking at the data

# Now we know column names
#Checking if any value is NaN

df.isnull().values.any()
def split_train_test(data, valid_ratio, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    valid_set_size = int(len(data) * valid_ratio)

    

    test_indices = shuffled_indices[:test_set_size]

    valid_indices = shuffled_indices[test_set_size:test_set_size+valid_set_size]

    train_indices = shuffled_indices[test_set_size+valid_set_size:]

    return data.iloc[train_indices], data.iloc[valid_indices], data.iloc[test_indices]



train_set, valid_set, test_set = split_train_test(df, 0.2, 0.2)

print(len(train_set), "train +", len(valid_set), "valid +", len(test_set), "test")
df =  train_set 
kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)

# Let's look at Sepal parameters 

plt.hist(df['SepalLengthCm'], **kwargs)

plt.hist(df['SepalWidthCm'], **kwargs);
# Now let's look at Petal parameters

plt.hist(df['PetalLengthCm'], **kwargs)

plt.hist(df['PetalWidthCm'], **kwargs);
df['Species'].value_counts() 

# And here we easily found 3 categories. 
plt.figure(figsize = (15,10))

sns.jointplot(df['SepalLengthCm'],df['SepalWidthCm'],kind="regg")

plt.show()

data = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

g = sns.PairGrid(data,diag_sharey = False,)





g.map_lower(sns.kdeplot,cmap="Blues_d")

g.map_upper(plt.scatter)

g.map_diag(sns.kdeplot,lw =3)



g = g.map(plt.scatter)
# Let's plot correlation map

f,ax=plt.subplots(figsize = (5,5))

sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.savefig('graph.png')

plt.show()
# Let's look at types of columns again

df.dtypes
# Let's split dataframe to two dataframes. One with "Species", which is 'object' type (obj_df). And another dataframe with numerical data (num_df)

obj_df = df.select_dtypes(include=['object', 'int64']).copy()

obj_df.head()
num_df = df.select_dtypes(include=['float64', 'int64']).copy()

num_df.head()
from sklearn.preprocessing import LabelBinarizer



lb_style = LabelBinarizer()

lb_results = lb_style.fit_transform(obj_df["Species"])

cat_df = pd.DataFrame(lb_results, columns=lb_style.classes_)

cat_df.head()
df = pd.concat([num_df, cat_df], axis=1, join='inner')

df.head()
df.dtypes
# We no longer need "Id". I'll just drop it.

df = df.drop(['Id'], axis=1)

df.head()
train = df.values

X = train[0::, 0:4]

y = train[0::, 4:7]
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



classifiers = [

    KNeighborsClassifier(3),

    #SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    #AdaBoostClassifier(),

    #GradientBoostingClassifier(),

    #GaussianNB(),

    #LinearDiscriminantAnalysis(),

    #QuadraticDiscriminantAnalysis(),

    #LogisticRegression()]

    ]



log_cols = ["Classifier", "Accuracy"]

log = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



train = df.values

X = train[0::, 0:4]

y = train[0::, 4:6]



acc_dict = {}



for train_index, test_index in sss.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    for clf in classifiers:

        print (clf)

        name = clf.__class__.__name__

        clf.fit(X_train, y_train)

        train_predictions = clf.predict(X_test)

        acc = accuracy_score(y_test, train_predictions)

        if name in acc_dict:

            acc_dict[name] += acc

        else:

            acc_dict[name] = acc



for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf] / 10.0

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



train = df.values

X = train[0::, 0:4]

y = train[0::, 4:5]



acc_dict = {}



for train_index, test_index in sss.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    for clf in classifiers:

        print (clf)

        name = clf.__class__.__name__

        clf.fit(X_train, y_train)

        train_predictions = clf.predict(X_test)

        acc = accuracy_score(y_test, train_predictions)

        if name in acc_dict:

            acc_dict[name] += acc

        else:

            acc_dict[name] = acc



for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf] / 10.0

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
log_cols = ["Classifier", "Accuracy"]

log = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



train = df.values

X = train[0::, 0:4]

y = train[0::, 5:6]



acc_dict = {}



for train_index, test_index in sss.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    for clf in classifiers:

        print (clf)

        name = clf.__class__.__name__

        clf.fit(X_train, y_train)

        train_predictions = clf.predict(X_test)

        acc = accuracy_score(y_test, train_predictions)

        if name in acc_dict:

            acc_dict[name] += acc

        else:

            acc_dict[name] = acc



for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf] / 10.0

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
log_cols = ["Classifier", "Accuracy"]

log = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=1022)



train = df.values

X = train[0::, 0:4]

y = train[0::, 6:7]



acc_dict = {}



for train_index, test_index in sss.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    for clf in classifiers:

        print (clf)

        name = clf.__class__.__name__

        clf.fit(X_train, y_train)

        train_predictions = clf.predict(X_test)

        acc = accuracy_score(y_test, train_predictions)

        if name in acc_dict:

            acc_dict[name] += acc

        else:

            acc_dict[name] = acc



for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf] / 10.0

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
# importing necessary libraries

from sklearn import datasets

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

 

# loading the iris dataset

iris = datasets.load_iris()





# X -> features, y -> label

X = iris.data

y = iris.target



print (len(X))



# dividing X, y into train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

 

# training a Naive Bayes classifier

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(X_train, y_train)

gnb_predictions = gnb.predict(X_test)

 

# accuracy on X_test

accuracy = gnb.score(X_test, y_test)

print (accuracy)

 

# creating a confusion matrix

cm = confusion_matrix(y_test, gnb_predictions)
classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]

    

log_cols = ["Classifier", "Accuracy"]

log = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



train = df.values



acc_dict = {}



for train_index, test_index in sss.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    for clf in classifiers:

        print (clf)

        name = clf.__class__.__name__

        clf.fit(X_train, y_train)

        train_predictions = clf.predict(X_test)

        acc = accuracy_score(y_test, train_predictions)

        if name in acc_dict:

            acc_dict[name] += acc

        else:

            acc_dict[name] = acc



for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf] / 10.0

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")