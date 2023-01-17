import numpy as np

import seaborn as sns

import functools

import pandas as pd

from sklearn import preprocessing, model_selection, ensemble, metrics, model_selection, neighbors, svm

%matplotlib inline
# Helper method I (tom) am introducing to make some code less imperative

def compose(*funcs):

    return lambda x: functools.reduce(lambda v, f: f(v), funcs, x)
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')



# data_train.sample(3)

data_train[0:4]
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train)
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"])
def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    # df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df



def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df



def format_name(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])

    return df    

    

def drop_features(df):

    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)



transform_features = compose(simplify_ages, simplify_cabins, simplify_fares, format_name, drop_features)



data_train = transform_features(data_train)

data_test = transform_features(data_test)

data_train.head()
sns.barplot(x="Age", y="Survived", hue="Sex", data=data_train)
sns.barplot(x="Cabin", y="Survived", hue="Sex", data=data_train)
features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']



def label_encoders(df_combined):

    for feature in features:

        yield feature, preprocessing.LabelEncoder().fit(df_combined[feature])



def feature_encoder(combined):

    def encode_features(df):

        for feature, le in label_encoders(combined):

            df[feature] = le.transform(df[feature])

        return df

    return encode_features



encode_features = feature_encoder(pd.concat([data_train[features], data_test[features]]))

data_test = encode_features(data_test)

data_train = encode_features(data_train)

data_train.head()
x_all = data_train.drop(['Survived', 'PassengerId'], axis=1)

y_all = data_train.Survived



num_test = 0.20

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_all, y_all, test_size=num_test, random_state=23)
# around 0.8

classifier = ensemble.RandomForestClassifier()

parameters = {'n_estimators': [4, 6, 9, 15], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



classifier = ensemble.BaggingClassifier(neighbors.KNeighborsClassifier(),

                                        max_samples=0.8, max_features=0.5)

parameters = { }



# mean is 0.6

classifier = svm.SVC()

parameters = {}



# mean is 0.83

classifier = ensemble.GradientBoostingClassifier()

parameters = {

    'max_depth': [3, 6, 10, 30],

    'n_estimators': [100, 300, 500, 1000, 2000]

}



# mean is 0.82

# classifier = ensemble.AdaBoostClassifier()

# parameters = {

#     # 'max_depth': [3, 6, 10]

# }
# Type of scoring used to compare parameter combinations

accuracy_scorer = metrics.make_scorer(metrics.accuracy_score)



# Run the grid search

grid_obj = model_selection.GridSearchCV(classifier, parameters, scoring=accuracy_scorer)

grid_obj = grid_obj.fit(x_train, y_train)



# Set the clf to the best combination of parameters

classifier = grid_obj.best_estimator_



# Fit the best algorithm to the data.

classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)

print(metrics.accuracy_score(y_test, predictions))
kf = model_selection.KFold(random_state=891, n_splits=10)

outcomes = []

for train_index, test_index in kf.split(x_train, y=y_train):

    x_train, x_test = x_all.values[train_index], x_all.values[test_index]

    y_train, y_test = y_all.values[train_index], y_all.values[test_index]

    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)

    outcomes.append(metrics.accuracy_score(y_test, predictions))

    

print("Outcomes: {0}".format(",".join(map(str, outcomes))))

mean_outcome = np.mean(outcomes)

print("Mean Accuracy: {0}".format(mean_outcome)) 
ids = data_test.PassengerId

predictions = classifier.predict(data_test.drop(['PassengerId'], axis=1))



output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('titanic-predictions.csv', index=False)