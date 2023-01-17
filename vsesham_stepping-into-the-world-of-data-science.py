""" importing required packages """

%matplotlib inline



""" packages for data manipulation, munging and merging """

import pandas as pd

import numpy as np



""" packages for visualiztion and exploratory analysis """

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)



""" configure visualization """



""" packages for running machine learning algorithms """



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
""" training and test files """

train_file = '../input/train.csv'

test_file = '../input/test.csv'

    

""" read training data into a pandas dataframe """

def read_training_data(filename):

    """ reading training data using pandas read_csv function """

    titanic_train = pd.read_csv(filename)

    return titanic_train



""" read test data into a pandas dataframe """

def read_test_data(filename):

    titanic_test = pd.read_csv(filename)

    return titanic_test



""" function to combine training and test dataframes """

def combine_dataframes(trainDF, testDF):

    combined = trainDF.append(testDF, ignore_index=True)

    return combined



train_df = read_training_data(train_file)

titanic_test = read_test_data(test_file)
def training_data_summary(train_df):

        """ print summary statistics for training dataset """

        print(train_df.describe())

        print(train_df.info())

        print(train_df.shape)

        print(train_df.head(5))

        print(train_df.tail(5))



""" output summary statistics on training data """

training_data_summary(train_df)
def fill_missing_values(df):

    """ filling missing values in Age column by computing the medain of Age """

    df['Age'] = df.Age.fillna(train_df.Age.median())

    df['Fare'] = df.Fare.fillna(train_df.Fare.median())

    df['Embarked'] = df.Embarked.fillna(method='ffill')

    return df



""" impute missing values """

train_df_filled = fill_missing_values(train_df)
def datatype_conversion(df):

    df['Sex'] = df['Sex'].astype('category')

    df['Pclass'] = df['Pclass'].astype('category')

    df['Embarked'] = df['Embarked'].astype('category')

    return df



""" perform datatype conversions """

titanic_df = datatype_conversion(train_df_filled)



""" converting the outcome variable of the dataset to int type  """

titanic_df["Survived"] = titanic_df['Survived'].astype('int')
def visualize_categories(df, **kwargs):

    row = kwargs.get('row',None)

    col = kwargs.get('col',None)

    hue = kwargs.get('hue',None)

    

    target_var = 'Survived'

    category_list = ['Sex','Pclass', 'Embarked']

    hue_var = 'Pclass'

    for cat_var in category_list:

        if cat_var != hue_var:

            plt.figure()

            sns.barplot(x=cat_var, y=target_var, hue=hue_var, data=df, ci=None)

        else:

            plt.figure()

            sns.barplot(x=cat_var, y=target_var, data=df, ci=None)



""" plotting categorial features against surivival rate """

visualize_categories(titanic_df)
def visualize_distribution(df, feature, **kwargs):

    row = kwargs.get('row',None)

    col = kwargs.get('col',None)



    g1 = sns.FacetGrid(df, row = row, col = col, aspect=2)

    g1.map(plt.hist, feature, bins=20, alpha=0.5)

    xlim_max = df[feature].max()

    g1.set(xlim=(0,xlim_max))

    g1.add_legend()



""" distribution plots """

visualize_distribution(titanic_df, feature='Age', col='Survived')

visualize_distribution(titanic_df, feature='Fare', col='Survived')

visualize_distribution(titanic_df, feature='Age', row='Pclass', col='Survived')

visualize_distribution(titanic_df, feature='Age', row='Sex', col='Survived')
def feature_conversion(df):

    df['Pclass'] = df['Pclass'].astype('int')

    df['Sex'] = df['Sex'].map({'male':0, 'female':1}).astype(int)

    df['Embarked'] = df['Embarked'].map({'C':0,'Q':1,'S':2}).astype(int)

    

    """ setting numerical age bands based on exploratory analysis observations"""

    df.loc[(df['Age'] <= 20),'Age'] = 0

    df.loc[(df['Age'] > 20) & (df['Age'] <= 28), 'Age'] = 1

    df.loc[(df['Age'] > 28) & (df['Age'] <= 38), 'Age'] = 2

    df.loc[(df['Age'] > 38) & (df['Age'] <= 80), 'Age'] = 3

    df.loc[(df['Age'] > 80), 'Age'] = 4

    

    df.loc[(df['Fare'] <= 7.91),'Fare'] = 0

    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454),'Fare'] = 1

    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31),'Fare'] = 2

    df.loc[(df['Fare'] > 31),'Fare'] = 3

    

    df['Age'] = df['Age'].astype('int')

    df['Fare'] = df['Fare'].astype('int')

    

    return df



train_df_prep = feature_conversion(titanic_df)
def drop_features(df):

    df = df.drop(['Ticket','Cabin','Name'], axis=1)

    return df



train_df_final = drop_features(train_df_prep)



""" perform operations on test dataset """

test_df_filled = fill_missing_values(titanic_test)

test_df_converted = datatype_conversion(test_df_filled)

test_df_prep = feature_conversion(titanic_test)

test_df_final = drop_features(test_df_prep)



combined = combine_dataframes(train_df_final, test_df_final)
def perform_logistic_regression(df_X, df_Y, test_df_X):

    logistic_regression = LogisticRegression()

    logistic_regression.fit(df_X, df_Y)

    pred_Y = logistic_regression.predict(test_df_X)

    accuracy = round(logistic_regression.score(df_X, df_Y) * 100,2)

    returnval = {'model':'Logistic Regression','accuracy':accuracy}

    return returnval
def perform_svc(df_X, df_Y, test_df_X):

    svc_clf = SVC()

    svc_clf.fit(df_X, df_Y)

    pred_Y = svc_clf.predict(test_df_X)

    accuracy = round(svc_clf.score(df_X, df_Y) * 100, 2)

    returnval = {'model':'SVC', 'accuracy':accuracy}

    return returnval
def perform_linear_svc(df_X, df_Y, test_df_X):

    svc_linear_clf = LinearSVC()

    svc_linear_clf.fit(df_X, df_Y)

    pred_Y = svc_linear_clf.predict(test_df_X)

    accuracy = round(svc_linear_clf.score(df_X, df_Y) * 100, 2)

    returnval = {'model':'LinearSVC', 'accuracy':accuracy}

    return returnval
def perform_rfc(df_X, df_Y, test_df_X):

    rfc_clf = RandomForestClassifier(n_estimators = 100 ,oob_score=True, max_features=None)

    rfc_clf.fit(df_X, df_Y)

    pred_Y = rfc_clf.predict(test_df_X)

    accuracy = round(rfc_clf.score(df_X, df_Y) * 100, 2)

    returnval = {'model':'RandomForestClassifier','accuracy':accuracy}

    return returnval
def perform_knn(df_X, df_Y, test_df_X):

    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(df_X, df_Y)

    pred_Y = knn.predict(test_df_X)

    accuracy = round(knn.score(df_X, df_Y) *100,2)

    returnval = {'model':'KNeighborsClassifier','accuracy':accuracy}

    return returnval
def perform_gnb(df_X, df_Y, test_df_X):

    gnb = GaussianNB()

    gnb.fit(df_X, df_Y)

    pred_Y = gnb.predict(test_df_X)

    accuracy = round(gnb.score(df_X, df_Y)*100,2)

    returnval = {'model':'GaussianNB','accuracy':accuracy}

    return returnval
def perform_dtree(df_X, df_Y, test_df_X):

    dtree = DecisionTreeClassifier()

    dtree.fit(df_X, df_Y)

    pred_Y = dtree.predict(test_df_X)

    accuracy = round(dtree.score(df_X, df_Y)*100,2)

    returnval = {'model':'DecisionTreeClassifier','accuracy':accuracy}

    return returnval
train_X = train_df_final.drop(['Survived','PassengerId'],axis=1)

train_Y = train_df_final['Survived']

test_X = test_df_final.drop('PassengerId', axis=1).copy()

    

lr_val = perform_logistic_regression(train_X, train_Y, test_X)

svc_val = perform_svc(train_X, train_Y, test_X)

svc_lin_val = perform_linear_svc(train_X, train_Y, test_X)

rfc_val = perform_rfc(train_X, train_Y, test_X)

knn_val = perform_knn(train_X, train_Y, test_X)

gnb_val = perform_gnb(train_X, train_Y, test_X)

dtree_val = perform_dtree(train_X, train_Y, test_X)

    

model_accuracies = pd.DataFrame()

model_accuracies = model_accuracies.append([lr_val,svc_val,svc_lin_val, rfc_val, knn_val, gnb_val, dtree_val])

cols = list(model_accuracies.columns.values)

cols = cols[-1:] + cols[:-1]

model_accuracies = model_accuracies[cols]

model_accuracies = model_accuracies.sort_values(by='accuracy')

print(model_accuracies)

plt.figure()

plt.xticks(rotation=90)

sns.barplot(x='model', y='accuracy', data=model_accuracies)
def write_to_csv(train_X, train_Y, test_df, test_X):

    rfc_clf = RandomForestClassifier(n_estimators = 100 ,oob_score=True, max_features=None)

    rfc_clf.fit(train_X, train_Y)

    pred_Y = rfc_clf.predict(test_X)

    pred_Y_list = pred_Y.tolist()

    test_X['Survived'] = pred_Y

    test_X['PassengerId'] = test_df['PassengerId']

    final_df = test_X[['PassengerId','Survived']]

    final_df.to_csv('passenger_survival.csv',sep=',',index=False)

    

write_to_csv(train_X, train_Y, test_df_final, test_X)