import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")
data_train.info()
data_test.info()
data_train.head()
sns.countplot(data_train.Survived)

plt.show()
df = data_train[(data_train.Age <= 13)]

plt.figure(figsize=(8,5))

sns.swarmplot("Sex", "Age", hue="Survived", data=df)

plt.show()
#df = data_train[(data_train.Age > 13) & (data_train.Age <= 18)]

plt.figure(figsize=(8,5))

sns.swarmplot("Sex", "Age", hue="Survived", data=data_train)

plt.show()
sns.heatmap(data_train.corr(), fmt="0.1f", annot=True)

plt.show()
plt.figure(figsize=(8,5))

sns.swarmplot("Pclass", "Fare", hue="Survived", data=data_train)

plt.show()
plt.figure(figsize=(8,5))

sns.swarmplot("Parch", "Age", hue="Survived", data=data_train)

plt.show()
plt.figure(figsize=(8,5))

sns.swarmplot("SibSp", "Age", hue="Survived", data=data_train)

plt.show()
sns.distplot(np.log1p(data_train.Fare))

plt.show()
def cabin_imputer(cabin):

    if cabin != "Unknown":

        return cabin[0]

    return cabin



def age_to_cat(age):

    if np.isnan(age):

        return "Unknown"

    elif age < 13:

        return "Kid"

    elif age <= 18:

        return "Teen"

    elif age > 60:

        return "Elder"

    else:

        return "Adult"



import re

def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if big_string.find(substring) != -1:

            return substring

    print(big_string)

    return np.nan



def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        return "Dr"

    else:

        return title



def clear_dataset(dataset):

    dataset["Family"] = dataset["Parch"] + dataset["SibSp"] + 1

    dataset["Alone"] = dataset["Family"] == 1

    dataset.Cabin = dataset.Cabin.fillna("Unknown")

    dataset["Cabin"] = dataset["Cabin"].apply(cabin_imputer)

    dataset.Embarked = dataset.Embarked.fillna("S")

    dataset.Fare = dataset.groupby(by="Pclass")["Fare"].apply(lambda x: x.fillna(x.mean()))

    dataset["Age_Cat"] = dataset.Age.apply(age_to_cat)

    dataset["Age"] = dataset.Age.fillna(dataset.Age.mean())

    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                    'Don', 'Jonkheer']

    dataset['Title']=dataset['Name'].map(lambda x: substrings_in_string(x, title_list))

    dataset['Title']=dataset.apply(replace_titles, axis=1)

    dataset["FareLog"] = dataset.Fare.apply(np.log1p)

    dataset["FarePerFamilyMember"] = dataset.Fare / dataset.Family

    dataset.drop(["Ticket","Name"], axis=1, inplace=True)

 

    return dataset
data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")

y_data = data_train.Survived

x_data_train = clear_dataset(data_train)

x_data_test = clear_dataset(data_test)
x_data_train.drop(["PassengerId", "Survived"], axis=1,inplace=True)

x_data_test.drop(["PassengerId"], axis=1,inplace=True)
# from sklearn.preprocessing import LabelEncoder

# encode_list = ["Sex","Cabin","Embarked","Age_Cat","Title"]

# for col in encode_list:

#     encoder = LabelEncoder()

#     x_data_train[col] = encoder.fit_transform(x_data_train[col])

#     x_data_test[col] = encoder.transform(data_test[col])

test_encoded = pd.get_dummies(x_data_test)

train_encoded = pd.get_dummies(x_data_train)

test_encoded= test_encoded.reindex(columns = train_encoded.columns, fill_value=0)
from sklearn.decomposition import PCA

pca = PCA(n_components = 2, whiten= True )  # whitten = normalize

x_pca = pca.fit_transform(train_encoded)

print("variance ratio: ", pca.explained_variance_ratio_)

print("sum: ",sum(pca.explained_variance_ratio_))
df = pd.DataFrame(np.stack([x_pca[:,0], x_pca[:,1], y_data], axis=1), columns=["p1", "p2", "Label"])

color = ["blue", "green"]

plt.figure(1,figsize=(9,6))

for each in set(y_data.unique()):

    plt.scatter(df.p1[df.Label == each],df.p2[df.Label == each],color = color[each - 1],label = each, alpha=0.5)

plt.legend()

plt.show()
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(train_encoded, y_data, test_size=0.25, random_state=42)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report



def get_metrics(y_test, y_predicted):  

    # true positives / (true positives+false positives)

    precision = precision_score(y_test, y_predicted, pos_label=None,

                                    average='weighted')             

    # true positives / (true positives + false negatives)

    recall = recall_score(y_test, y_predicted, pos_label=None,

                              average='weighted')

    

    # harmonic mean of precision and recall

    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    

    # true positives + true negatives/ total

    accuracy = accuracy_score(y_test, y_predicted)

    return accuracy, precision, recall, f1
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=300, max_depth=6,max_features=11,criterion="gini",n_jobs=-1, random_state=42)

clf.fit(x_train, y_train)



y_predicted = clf.predict(x_val)

accuracy, precision, recall, f1 = get_metrics(y_val, y_predicted)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
from sklearn.metrics import confusion_matrix

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
y_predicted = clf.predict(x_val)

cm = confusion_matrix(y_val, y_predicted)

plt.figure(1, figsize=(6,4))

plot_confusion_matrix(cm, classes = set(y_data.unique()))

plt.show()
def plot_feature_importances(clf, X_train, y_train=None, 

                             top_n=10, figsize=(8,8), print_table=False, title="Feature Importances"):

    '''

    plot feature importances of a tree-based sklearn estimator

    

    Note: X_train and y_train are pandas DataFrames

    

    Note: Scikit-plot is a lovely package but I sometimes have issues

              1. flexibility/extendibility

              2. complicated models/datasets

          But for many situations Scikit-plot is the way to go

          see https://scikit-plot.readthedocs.io/en/latest/Quickstart.html

    

    Parameters

    ----------

        clf         (sklearn estimator) if not fitted, this routine will fit it

        

        X_train     (pandas DataFrame)

        

        y_train     (pandas DataFrame)  optional

                                        required only if clf has not already been fitted 

        

        top_n       (int)               Plot the top_n most-important features

                                        Default: 10

                                        

        figsize     ((int,int))         The physical size of the plot

                                        Default: (8,8)

        

        print_table (boolean)           If True, print out the table of feature importances

                                        Default: False

        

    Returns

    -------

        the pandas dataframe with the features and their importance

        

    Author

    ------

        George Fisher

    '''

    

    __name__ = "plot_feature_importances"

    

    import pandas as pd

    import numpy  as np

    import matplotlib.pyplot as plt

    

    from xgboost.core     import XGBoostError

    from lightgbm.sklearn import LightGBMError

    

    try: 

        if not hasattr(clf, 'feature_importances_'):

            clf.fit(X_train.values, y_train.values.ravel())



            if not hasattr(clf, 'feature_importances_'):

                raise AttributeError("{} does not have feature_importances_ attribute".

                                    format(clf.__class__.__name__))

                

    except (XGBoostError, LightGBMError, ValueError):

        clf.fit(X_train.values, y_train.values.ravel())

            

    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    

    feat_imp['feature'] = X_train.columns

    feat_imp.sort_values(by='importance', ascending=False, inplace=True)

    feat_imp = feat_imp.iloc[:top_n]

    

    feat_imp.sort_values(by='importance', inplace=True)

    feat_imp = feat_imp.set_index('feature', drop=True)

    feat_imp.plot.barh(title=title, figsize=figsize)

    plt.xlabel('Feature Importance Score')

    plt.show()

    

    if print_table:

        from IPython.display import display

        print("Top {} features in descending order of importance".format(top_n))

        display(feat_imp.sort_values(by='importance', ascending=False))

        

    return feat_imp
a = plot_feature_importances(clf, x_train, y_train, top_n=x_train.shape[1], title=clf.__class__.__name__)
y_predicted = clf.predict(x_train)

accuracy, precision, recall, f1 = get_metrics(y_train, y_predicted)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
from xgboost import XGBClassifier

clf = XGBClassifier(n_estimators= 300, learning_rate=0.3, max_depth=4)

clf.fit(x_train, y_train)

y_predicted = clf.predict(x_val)

accuracy, precision, recall, f1 = get_metrics(y_val, y_predicted)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
cm = confusion_matrix(y_val, y_predicted)

plt.figure(1, figsize=(6,4))

plot_confusion_matrix(cm, classes = set(y_data.unique()))

plt.show()
a = plot_feature_importances(clf, x_train, y_train, top_n=x_train.shape[1], title=clf.__class__.__name__)
y_predicted = clf.predict(x_train)

accuracy, precision, recall, f1 = get_metrics(y_train, y_predicted)

print("train accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=2.0, solver="newton-cg", penalty="l2", n_jobs=-1)

clf.fit(x_train, y_train)

y_predicted = clf.predict(x_val)

accuracy, precision, recall, f1 = get_metrics(y_val, y_predicted)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
cm = confusion_matrix(y_val, y_predicted)

plt.figure(1, figsize=(6,4))

plot_confusion_matrix(cm, classes = set(y_data.unique()))

plt.show()
y_predicted = clf.predict(x_train)

accuracy, precision, recall, f1 = get_metrics(y_train, y_predicted)

print("train accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

clf = GaussianNB()

clf.fit(x_train, y_train)

y_predicted = clf.predict(x_val)

accuracy, precision, recall, f1 = get_metrics(y_val, y_predicted)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
cm = confusion_matrix(y_val, y_predicted)

plt.figure(1, figsize=(6,4))

plot_confusion_matrix(cm, classes = set(y_data.unique()))

plt.show()
y_predicted = clf.predict(x_train)

accuracy, precision, recall, f1 = get_metrics(y_train, y_predicted)

print("train accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
clf = BernoulliNB(alpha=0.2)

clf.fit(x_train, y_train)

y_predicted = clf.predict(x_val)

accuracy, precision, recall, f1 = get_metrics(y_val, y_predicted)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
cm = confusion_matrix(y_val, y_predicted)

plt.figure(1, figsize=(6,4))

plot_confusion_matrix(cm, classes = set(y_data.unique()))

plt.show()
y_predicted = clf.predict(x_train)

accuracy, precision, recall, f1 = get_metrics(y_train, y_predicted)

print("train accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
from sklearn.svm import SVC

clf = SVC(C=40)

clf.fit(x_train, y_train)

y_predicted = clf.predict(x_val)

accuracy, precision, recall, f1 = get_metrics(y_val, y_predicted)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
cm = confusion_matrix(y_val, y_predicted)

plt.figure(1, figsize=(6,4))

plot_confusion_matrix(cm, classes = set(y_data.unique()))

plt.show()
y_predicted = clf.predict(x_train)

accuracy, precision, recall, f1 = get_metrics(y_train, y_predicted)

print("train accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
from sklearn.model_selection import cross_val_score

clf = RandomForestClassifier(n_estimators=100, max_depth=6,max_features=11, min_samples_leaf=0.0001,criterion="gini",n_jobs=-1, random_state=42)

accuracies = cross_val_score(clf, train_encoded, y_data, cv=5, scoring="accuracy")

print("CV accuracy", accuracies.mean())
clf = clf = LogisticRegression(C=0.9, solver="newton-cg", penalty="l2", n_jobs=-1)

accuracies = cross_val_score(clf, train_encoded, y_data, cv=5, scoring="accuracy")

print("CV accuracy", accuracies.mean())
submission = pd.read_csv("../input/gender_submission.csv")
clf = RandomForestClassifier(n_estimators=100, max_depth=6,max_features=11, min_samples_leaf=0.0001,criterion="gini",n_jobs=-1, random_state=42)

clf.fit(train_encoded, y_data)



test_preds = clf.predict(test_encoded)



submission.Survived = test_preds

submission.to_csv('rf_submission.csv', index=False)