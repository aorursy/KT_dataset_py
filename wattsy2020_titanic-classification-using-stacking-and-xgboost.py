import re

import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold, RandomizedSearchCV

from scipy.stats import uniform

from keras.layers import Input, Dense, Dropout

from keras.models import Model

from keras.optimizers import Adam
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

full_data = [train, test]



# Gives the length of the name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)

# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

    # Create new feature IsAlone from FamilySize

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

    # Remove all NULLS in the Embarked column

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    

    # Remove all NULLS in the Fare column and create a new feature CategoricalFare

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)



# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)



# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

    

    # Group all non-common titles into one single grouping "Rare"

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Lady', 'Countess'], 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;

    

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test_id = test['PassengerId']

test  = test.drop(drop_elements, axis = 1)



X_train = train.drop(['Survived'], axis = 1)

Y_train = train['Survived']
def accuracy(model):

    return np.sum(model.predict(X_train) == Y_train)/len(Y_train)



def output_results(name, predictions):

    output = (pd.DataFrame([list(np.array(test_id)), list(predictions)])

              .T

              .rename(columns={0: "PassengerId", 1: "Survived"}))

    output['Survived'] = output['Survived'].astype('int')

    output.to_csv(name + '.csv', index = False)
# Wrap the model in an adaptor class to make it simillar to an sklearn model

class NeuralNet:

    def __init__(self, X_train, Y_train, dropout_ratio = 0.7, epochs = 200):

        X_input = Input((X_train.shape[1],))

        X = Dense(200, activation="relu")(X_input)

        X = Dropout(dropout_ratio)(X)

        X = Dense(200, activation="relu")(X)

        X = Dropout(dropout_ratio)(X)

        X = Dense(200, activation="relu")(X)

        X = Dropout(dropout_ratio)(X)

        X = Dense(200, activation="relu")(X)

        X = Dropout(dropout_ratio)(X)

        X = Dense(1, activation="sigmoid")(X)



        self.model = Model(inputs=X_input, outputs=X, name='model')

        self.model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

        self.model.fit(x=X_train, y=Y_train, epochs=epochs, verbose=0)

    

    def predict(self, X):

        return (self.model.predict(X) >= 0.5)[:, 0]

        

    def predict_proba(self, X):

        pred = self.model.predict(X)[:, 0]

        result = np.zeros((len(pred), 2))

        for i in range(len(pred)):

            result[i, :] = 1 - pred[i], pred[i]

        return result
def MLP(X_train, Y_train):

    mlp = MLPClassifier(hidden_layer_sizes = (200, 100, 50),

                        activation = 'relu',

                        solver = 'adam',

                        learning_rate_init = 0.001)

    mlp.fit(X_train, Y_train)

    return mlp
def random_forest(X_train, Y_train):

    forest = RandomForestClassifier(n_estimators=200,

                                    min_samples_split=4,

                                    min_samples_leaf=3)

    forest.fit(X_train, Y_train)

    return forest
def SVM(X_train, Y_train, kernel = 'rbf'):

    svm = SVC(C = 1, kernel = kernel, probability = True, gamma='auto')

    svm.fit(X_train, Y_train)

    return svm
def KNN(X_train, Y_train):

    knn = KNeighborsClassifier(n_neighbors = 10)

    knn.fit(X_train, Y_train)

    return knn
def AdaBoost(X_train, Y_train):

    ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=6, min_samples_split=4, min_samples_leaf=3),

                             n_estimators = 100)

    ada.fit(X_train, Y_train)

    return ada
def GradBoost(X_train, Y_train):

    grad = GradientBoostingClassifier(n_estimators = 100,

                                     min_samples_split = 4,

                                     min_samples_leaf = 3)

    grad.fit(X_train, Y_train)

    return grad
# Define Lambdas to create each of the models

models = np.array([lambda X, Y: MLP(X, Y),

         lambda X, Y: random_forest(X, Y),

         lambda X, Y: SVM(X, Y, kernel = 'rbf'),

         lambda X, Y: SVM(X, Y, kernel = 'linear'),

         lambda X, Y: SVM(X, Y, kernel = 'sigmoid'),

         lambda X, Y: KNN(X, Y),

         lambda X, Y: AdaBoost(X, Y),

         lambda X, Y: GradBoost(X, Y),

         lambda X, Y: NeuralNet(X, Y, dropout_ratio=1, epochs=300)]) # I remove the dropout of the NN to increase variance and hopefully have it be less correlated with the other models

model_names = np.array(["MLP", "Random Forest", "SVM RBF", "SVM linear", 

                        "SVM sigmoid", "KNN", "AdaBoost", "GradBoost", "NeuralNet"])
num_features = len(X_train.columns)

feature_names = X_train.columns

working_models = [1, 6, 7]



_, subplots = plt.subplots(3, figsize=(15,15))

for title, model, subplt in zip(model_names[working_models], models[working_models], subplots):

    model = model(X_train, Y_train)

    subplt.bar(x=range(num_features), height=model.feature_importances_, tick_label=feature_names)

    subplt.set_title(title)

    subplt.set_ylim([0, 1])
# train and get predictions of each model on the test set

trained_models = np.array([model_lambda(X_train, Y_train) for model_lambda in models])

predictions = [model.predict_proba(test)[:, 1] for model in trained_models]

predictions = pd.DataFrame(data=np.array(predictions).T, columns=model_names)



# plot a heatmap of the corellations

plt.figure(figsize=(14,12))

plt.title('Correlation of Predictions', y=1.05, size=15)

sns.heatmap(predictions.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
trained_models = np.delete(trained_models, [4])

models = np.delete(models, [4])

model_names = np.delete(model_names, [4])
# Predict the entire data set using the out of fold predictions of a model

def get_fold_predictions(model_lambda, X_train, Y_train, n_folds):

    predictions = np.zeros(len(Y_train))

    folds = KFold(n_splits = n_folds).split(X_train, Y_train)

    for train_index, test_index in folds:

        model = model_lambda(X_train.loc[train_index, :], Y_train[train_index])

        predictions[test_index] = model.predict_proba(X_train.loc[test_index, :])[:, 1]

    return predictions



# For each model get it's predictions

predictions = [get_fold_predictions(model, X_train, Y_train, 5) for model in models]



# Build the stack train dataset

stack_train = pd.DataFrame(data=np.array(predictions).T, columns=model_names)
XGBoost = xgb.XGBClassifier()

distributions = {

    'max_depth': range(1, 4), # Depth of the trees should be on the lower side given there are only 8 features

    'learning_rate': uniform(loc=0, scale = 0.5), # might be better to use a logarithmic distribution to sample more lower points e.g. .01

    'n_estimators': range(1000),

    'reg_alpha': uniform(loc=0, scale=5),

    'reg_lambda': uniform(loc=0, scale=5),

    'booster': ['gbtree', 'gblinear', 'dart'],

    'min_child_weight': range(20),

    'subsample': uniform(loc=0, scale=1)

}

grid = RandomizedSearchCV(XGBoost, distributions, n_iter=100, cv=KFold(n_splits=3), verbose=0)

stack_model = grid.fit(stack_train, Y_train)

print("Accuracy: {}".format(np.sum(stack_model.predict(stack_train) == Y_train)/len(Y_train)))

stack_model.best_params_
test_predictions = [model.predict_proba(test)[:, 1] for model in trained_models]

stack_test = pd.DataFrame(data=np.array(test_predictions).T, columns=model_names)

ensemble_predictions = stack_model.predict(stack_test)

output_results('ensemble', ensemble_predictions)