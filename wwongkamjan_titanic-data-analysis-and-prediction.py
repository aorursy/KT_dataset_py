import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import re



# file path for train and test data

train_path = "../input/titanic/train.csv"

test_path = "../input/titanic/test.csv"





# read the files into variables

train_data = pd.read_csv(train_path, index_col=False)

test_data = pd.read_csv(test_path, index_col=False)





# delete tickets from columns

cols = list(train_data.columns)

train_data = train_data[cols[1:8] +cols[9:]]

cols = list(test_data.columns)

test_data = test_data[cols[1:7] +cols[8:]]
# fill null values in Age, Fare, Embarked 

train_data = train_data.astype({'Age':'float'}) 

test_data = test_data.astype({'Age':'float'}) 

train_data['Age'].fillna(train_data['Age'].median(), inplace = True)

train_data['Fare'].fillna(train_data['Fare'].median(), inplace = True)

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace = True)

test_data['Age'].fillna(test_data['Age'].median(), inplace = True)

test_data['Fare'].fillna(test_data['Fare'].median(), inplace = True)

test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace = True)
# add Has_Cabin as a new column, will be in use later when we train a model

train_data['Has_Cabin'] = train_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test_data['Has_Cabin'] = test_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
# function to get Mr. Miss etc. from Name

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""
# call the above function

train_data['Title'] = train_data['Name'].apply(get_title)

test_data['Title'] = test_data['Name'].apply(get_title)
# replace repetetive titles which have the same meaning (ex. Mlle and Miss)

train_data['Title'] = train_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')

train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')

train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')



test_data['Title'] = test_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')

test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')

test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')
# add no_of_fam column to be number of member in family -> which we can retrieve the number from SibSp and Parch

train_data['no_of_fam'] = train_data['SibSp'] + train_data['Parch']

test_data['no_of_fam'] = test_data['SibSp'] + test_data['Parch']

drop_column = ['SibSp','Parch','Name','Cabin']

train_data.drop(drop_column, axis=1, inplace = True)

test_data.drop(drop_column, axis=1, inplace = True)
# add Age_Band column to classify age into 4 groups

train_data.loc[ train_data['Age'] <= 16, 'Age_Band'] = 0

train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 32), 'Age_Band'] = 1

train_data.loc[(train_data['Age'] > 32) & (train_data['Age'] <= 48), 'Age_Band'] = 2

train_data.loc[(train_data['Age'] > 48) & (train_data['Age'] <= 64), 'Age_Band'] = 3

train_data.loc[ train_data['Age'] > 64, 'Age_Band'] = 4 



test_data.loc[ test_data['Age'] <= 16, 'Age_Band'] = 0

test_data.loc[(test_data['Age'] > 16) & (test_data['Age'] <= 32), 'Age_Band'] = 1

test_data.loc[(test_data['Age'] > 32) & (test_data['Age'] <= 48), 'Age_Band'] = 2

test_data.loc[(test_data['Age'] > 48) & (test_data['Age'] <= 64), 'Age_Band'] = 3

test_data.loc[ test_data['Age'] > 64, 'Age_Band'] = 4 
g = sns.distplot(train_data["Fare"], color="m", label="Skewness : %.2f"%(train_data["Fare"].skew()))

g = g.legend(loc="best")



# apply log to Fare

train_data['Fare'] = train_data['Fare'].map(lambda i : np.log(i) if i>0 else 0)
g = sns.distplot(train_data["Fare"], color="b", label="Skewness : %.2f"%(train_data["Fare"].skew()))

g = g.legend(loc="best")
g = sns.distplot(test_data["Fare"], color="m", label="Skewness : %.2f"%(test_data["Fare"].skew()))

g = g.legend(loc="best")



# apply log to Fare

test_data['Fare'] = test_data['Fare'].map(lambda i : np.log(i) if i>0 else 0)
g = sns.distplot(test_data["Fare"], color="b", label="Skewness : %.2f"%(test_data["Fare"].skew()))

g = g.legend(loc="best")
total_survived_passengers = train_data['Survived'].loc[(train_data['Survived']==1)].count()

total_survived_female_passengers =train_data['Survived'].loc[(train_data['Sex']=="female") & (train_data['Survived']==1)].count()

total_survived_male_passengers =train_data['Survived'].loc[(train_data['Sex']=="male") & (train_data['Survived']==1)].count()



# Bar chart 

sns.set_palette("RdBu", n_colors=2)

sns.barplot(x=['male','female'], y=[total_survived_male_passengers/total_survived_passengers, total_survived_female_passengers/total_survived_passengers])



# Add label for vertical axis

plt.ylabel("Survival Rate by Gender")

plt.xlabel("Gender")
print("by all survivors, being female ", total_survived_female_passengers/total_survived_passengers)

print("by all survivors, being male ", total_survived_male_passengers/total_survived_passengers)
# Bar chart 

sns.set_palette("RdBu", n_colors=2)

sns.barplot(x='Sex', y='Survived', data=train_data);



# Add label for vertical axis

plt.ylabel("Survival Rate in Male and Female")

plt.xlabel("Gender")
total_female_passengers = train_data['Survived'].loc[(train_data['Sex']=="female")].count()

total_survived_female_passengers =train_data['Survived'].loc[(train_data['Sex']=="female") & (train_data['Survived']==1)].count()

print("by all female passengers, being survived", total_survived_female_passengers/total_female_passengers)

total_male_passengers = train_data['Survived'].loc[(train_data['Sex']=="male")].count()

total_survived_male_passengers =train_data['Survived'].loc[(train_data['Sex']=="male") & (train_data['Survived']==1)].count()

print("by all male passengers, being survived", total_survived_male_passengers/total_male_passengers)



#cross tab 

pd.crosstab(train_data['Title'], train_data['Survived']).apply(lambda r: r/r.sum(), axis=1)
#cross tab 

pd.crosstab(train_data['Age_Band'], train_data['Survived']).apply(lambda r: r/r.sum(), axis=1)
pd.crosstab(train_data['no_of_fam'], train_data['Survived']).apply(lambda r: r/r.sum(), axis=1)
pd.crosstab(train_data['Has_Cabin'], train_data['Survived']).apply(lambda r: r/r.sum(), axis=1)
pd.crosstab(train_data['Sex'], train_data['Survived']).apply(lambda r: r/r.sum(), axis=1)
from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OneHotEncoder





cols = list(train_data.columns)

X = train_data[['Sex','Age_Band',]].values

y = train_data[['Survived']].values.ravel()



# label Sex and Age_Range to be int!

Le = LabelEncoder()

X[:,0] = Le.fit_transform (X[:,0])

X[:,1] = Le.fit_transform (X[:,1])



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)



model = svm.SVC()

model.fit(X_train, y_train)



predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)



print("predictions:", predictions)

print("actual: ", y_test)

print("accuracy: ", acc)
from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler



train_data['Sex'] = train_data['Sex'].astype('category')

train_data['Pclass'] = train_data['Pclass'].astype('category')

train_data['Embarked'] = train_data['Embarked'].astype('category')

train_data['Title'] = train_data['Title'].astype('category')



# label Sex and Age_Range to be int!

train_data['Sex'] = train_data['Sex'].cat.codes

train_data['Pclass'] = train_data['Pclass'].cat.codes

train_data['Embarked'] = train_data['Embarked'].cat.codes

train_data['Title'] = train_data['Title'].cat.codes





train_data = pd.get_dummies(train_data, columns=['Sex'], prefix = ['Sex'])

train_data = pd.get_dummies(train_data, columns=['Pclass'], prefix = ['Pclass'])

train_data = pd.get_dummies(train_data, columns=['Embarked'], prefix = ['Embarked'])

train_data = pd.get_dummies(train_data, columns=['Title'], prefix = ['Title'])



test_data['Sex'] = test_data['Sex'].astype('category')

test_data['Pclass'] = test_data['Pclass'].astype('category')

test_data['Embarked'] = test_data['Embarked'].astype('category')

test_data['Title'] = test_data['Title'].astype('category')



# label Sex and Age_Range to be int!

test_data['Sex'] = test_data['Sex'].cat.codes

test_data['Pclass'] = test_data['Pclass'].cat.codes

test_data['Embarked'] = test_data['Embarked'].cat.codes

test_data['Title'] = test_data['Title'].cat.codes





test_data = pd.get_dummies(test_data, columns=['Sex'], prefix = ['Sex'])

test_data = pd.get_dummies(test_data, columns=['Pclass'], prefix = ['Pclass'])

test_data = pd.get_dummies(test_data, columns=['Embarked'], prefix = ['Embarked'])

test_data = pd.get_dummies(test_data, columns=['Title'], prefix = ['Title'])



X = train_data[['Fare', 'Has_Cabin', 'no_of_fam', 'Age_Band',

       'Sex_0', 'Sex_1', 'Pclass_0', 'Pclass_1', 'Pclass_2', 'Embarked_0',

       'Embarked_1', 'Embarked_2', 'Title_0', 'Title_1', 'Title_2', 'Title_3',

       'Title_4']].values

y = train_data[['Survived']].values.ravel()



# X_test = test_data[['Fare', 'Has_Cabin', 'no_of_fam', 'Age_Band',

#        'Sex_0', 'Sex_1', 'Pclass_0', 'Pclass_1', 'Pclass_2', 'Embarked_0',

#        'Embarked_1', 'Embarked_2', 'Title_0', 'Title_1', 'Title_2', 'Title_3',

#        'Title_4']].values



scaler = MinMaxScaler(feature_range=(0,1))

scaled_X = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(scaled_X,y, test_size=0.3)



model = svm.SVC()

model.fit(X_train, y_train)





predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)



print("predictions:", predictions)

print("actual: ", y_test)

print("accuracy: ", acc)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential #what is sequential? most simplist type good for one tensor input and output

from tensorflow.keras.layers import Activation, Dense

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.metrics import categorical_crossentropy



X_train = X_train.astype('float') 

y_train = y_train.astype('float')



model = Sequential([

    Dense(units=16, input_shape=(17,), activation='sigmoid'),

    Dense(units=2, activation='sigmoid')

])



model.compile(optimizer=Adam(learning_rate=0.02), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x=X_train, y= y_train,validation_split=0.2, batch_size=15, epochs=60, shuffle=True, verbose=0)



predictions = model.predict(x=X_test

    , batch_size=15

    , verbose=0)



rounded_predictions = np.argmax(predictions, axis=-1)



acc = accuracy_score(y_test, rounded_predictions)



print("predictions:", rounded_predictions)

print("actual: ", y_test)

print("accuracy: ", acc)
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

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

            horizontalalignment="center",

            color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
%matplotlib inline

from sklearn.metrics import confusion_matrix

import itertools

import matplotlib.pyplot as plt



cm = confusion_matrix(y_true=y_test, y_pred=rounded_predictions)

cm_plot_labels = ['Not Survived','Survived']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

neigh_clf = KNeighborsClassifier(n_neighbors=4)

log_clf = LogisticRegression()

svm_clf = SVC()

voting_clf = VotingClassifier(

estimators=[('lr', log_clf), ('svc', svm_clf),('knn', neigh_clf)],

voting='hard'

)

voting_clf.fit(X_train, y_train)



for clf in (log_clf, svm_clf,neigh_clf, voting_clf):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
#SVM 



model = svm.SVC()

model.fit(scaled_X, y)



X_test = test_data[['Fare', 'Has_Cabin', 'no_of_fam', 'Age_Band',

       'Sex_0', 'Sex_1', 'Pclass_0', 'Pclass_1', 'Pclass_2', 'Embarked_0',

       'Embarked_1', 'Embarked_2', 'Title_0', 'Title_1', 'Title_2', 'Title_3',

       'Title_4']].values



scaled_X_test = scaler.fit_transform(X_test)



predictions = model.predict(scaled_X_test)



print("predictions:", predictions)


model = Sequential([

    Dense(units=16, input_shape=(17,), activation='sigmoid'),

    Dense(units=2, activation='sigmoid')

])



model.compile(optimizer=Adam(learning_rate=0.02), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x=scaled_X, y= y,validation_split=0.2, batch_size=15, epochs=60, shuffle=True, verbose=0)



predictions = model.predict(x=scaled_X_test

    , batch_size=15

    , verbose=0)

rounded_predictions = np.argmax(predictions, axis=-1)



print("predictions:", rounded_predictions)
neigh_clf = KNeighborsClassifier(n_neighbors=4)

log_clf = LogisticRegression()

svm_clf = SVC()

voting_clf = VotingClassifier(

estimators=[('lr', log_clf), ('svc', svm_clf),('knn', neigh_clf)],

voting='hard'

)

voting_clf.fit(scaled_X, y)

    

predictions = voting_clf.predict(scaled_X_test)

print("predictions:", predictions)