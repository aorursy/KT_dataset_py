#Author: Caleb Woy

!python --version

# preprocessing

import pandas as pd # data manipulation

import numpy as np # linear algebra

from numpy.random import seed # setting random seeds

from scipy.stats import kurtosis, skew, t # checking distributions, t score

import matplotlib.pyplot as plt # plotting

import math # checking for NaNs

import seaborn as sb # plotting

import operator # indexing for max

# modelling

import scipy.spatial.distance as sp # kNN distance

from sklearn.model_selection import train_test_split # validation

from sklearn.model_selection import RandomizedSearchCV # tuning hyperparams 

from sklearn.model_selection import GridSearchCV # tuning hyperparams 

from sklearn.linear_model import LogisticRegression # Logistic Regression

from sklearn import tree # Decision tree

from sklearn.ensemble import RandomForestClassifier as rfc # Random Forest

from sklearn.ensemble import GradientBoostingClassifier as gbc # GBC

from sklearn.ensemble import AdaBoostClassifier as ada_boost # generic boosting

from sklearn.neighbors import KNeighborsClassifier as knn # K nearest neighbors

from sklearn.naive_bayes import MultinomialNB as mnb # Naive Bayes

from sklearn.naive_bayes import GaussianNB as gnb # Naive Bayes

import tensorflow as tf # neural networks

from keras.wrappers.scikit_learn import KerasClassifier # neural networks

from keras.models import Sequential  # neural networks

from keras.layers import Dense, Activation, Dropout  # neural networks
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

# appending 

data = train_data.append(test_data)

data.head()
data.dtypes
data['Pclass'] = data['Pclass'].astype('category')

data['Cabin'] = data['Cabin'].astype('category')

data['Embarked'] = data['Embarked'].astype('category')

data['Sex'] = data['Sex'].astype('category')

data.dtypes
"""

Creating interaction plots between two categorical variables.

"""

def interaction_plot(hue_lab, x_lab, data):

    # Grouping by the hue_group, then counting by the x_lab group

    hue_group = data.groupby([hue_lab], sort=False)

    counts = hue_group[x_lab].value_counts(normalize=True, sort=False)



    # Creating the percentage vector to measure the frequency of each type

    data = [

        {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

        (hue, x), percentage in dict(counts).items()

    ]



    # Creating and plotting the new dataframe 

    df = pd.DataFrame(data)

    p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

    p.set_xticklabels(rotation=90)

    p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')

 



"""

Extracting the title of an individual from their full name.

"""

def extract_title(string_name, titles):

    for title in titles:

        if (title in string_name):

            break

    return title



"""

Extracting the last name of an individual from their full name.

"""

def extract_last_name(string_name):

    return string_name.split(',')[0]



"""

Creating box plots between x numeric and y categorical.

"""

def box_plot(x, y, data):

    ax = sb.boxplot(x=data[x], y=data[y])

    ax.set_xticklabels(ax.get_xticklabels(),rotation=-90)



"""

Return a float of the suspected family's mortality rate. -1 of unknown or solo

passenger, else return rate of mortality.

"""

def get_fam_mort(last_name, data, row):

    mort = 0.0

    if row['Parch'] == 0 and row['SibSp'] == 0:

        # unknown

        mort = -1 

    else:

        fam_rows = data[(data['LastName'] == last_name) & 

                        (data['Parch'] > 0) & (data['SibSp'] > 0)]

        if len(fam_rows) == 0:

            # unknown

            mort = -1

        else:

            # mortality rate

            mort = len(fam_rows[fam_rows['Survived'] == 0]) / len(fam_rows)

    return mort





"""

Imputes the missing age values based on the mean age value of the sub group 

matching the class of the individual.

"""

def impute_age_by_class(Pclass, data, row):

    if math.isnan(row['Age']):

        return np.mean(data[data['Pclass'] == Pclass]['Age'])

    else:

        return row['Age']

    

"""

Imputes the missing fare values based on the mean fare value of the sub group 

matching the class of the individual. Also replaces fare values equal 

to 0 that're believed to be erroneous.

"""

def impute_fare_by_class(Pclass, data, row):

    if math.isnan(row['Fare']) or row['Fare'] == 0:

        return np.mean(data[data['Pclass'] == Pclass]['Fare'])

    else:

        return row['Fare']



"""

Min-max scales the column.

"""

def min_max(feature_name, data):

    min_d, max_d = min(data[feature_name]), max(data[feature_name])

    return data.apply(lambda x: (x[feature_name] - min_d) / (max_d - min_d),

                    axis = 1)
# setting the feature name we're examining so that we don't have to re-type it

feature_name = "Pclass"

# getting the count of each factor level of Pclass

data.groupby(feature_name).count()["PassengerId"]
data[feature_name].isnull().sum()
interaction_plot("Survived", feature_name, data)
# setting the feature name we're examining so that we don't have to re-type it

feature_name = "Name"

# Checking for missing values

data[feature_name].isnull().sum()
# list of all personal titles in our data set

titles = ["Jonkheer.", "Countess.","Lady.","Sir.","Dona.", "Mme.", "Master.", 

          "Rev.", "Don.", "Dr.", "Major.", "Col.", "Capt.", "Mr.", "Mrs.", 

          "Miss.", "Ms.", "Mlle."]

# calling .apply() with the argument axis=1 will apply the lambda function

# argument to each row of the column. The variable x in the lambda represents a 

# row. The funcion extract_title is defined above in our function definitions

# cell. We'll save this as a new feature on our DataFrame called 'Title'

data['Title'] = data.apply(lambda x:

                                extract_title(x[feature_name], titles), axis=1)

# verifying

data.head()
# this apply call is very similar to the previous. extract_last_name is 

# also defined above in our function definitions cell. We'll save this as a 

# new feature on our DataFrame called 'LastName'

data['LastName'] = data.apply(lambda x:

                                extract_last_name(x[feature_name]), axis=1)

# verifying

data.head()
data[['LastName']].describe()
# this apply call is very similar to the previous. get_fam_mort is 

# also defined above in our function definitions cell. We'll save this as a 

# new feature on our DataFrame called 'FamMort'

data['FamMort'] = data.apply(lambda x:

                        get_fam_mort(x['LastName'], 

                        data[~data["Survived"].isna()], 

                        x),

                        axis=1)

# Individuals that are likely single passengers are encoded as '-1.0', all

# others are encoded as the percentage of likely family members in the 

# training data that perished. We don't want the '-1.0' to have numeric 

# meaning so we're setting this column as a category.

data['FamMort'] = data['FamMort'].astype('category')



# verifying

data.head()
interaction_plot("Survived", 'Title', data)
box_plot("FamMort","Survived", data)
data[['FamMort']].describe()
# setting the feature name we're examining so that we don't have to re-type it

feature_name = "Age"

# Checking for missing values

data[feature_name].isnull().sum()
correlation_matrix = data.corr().round(2)

plt.figure(figsize=(10,8))

sb.heatmap(data=correlation_matrix, annot=True, center=0.0, cmap='coolwarm')
box_plot("Pclass","Age", data)
# The impute_age_by_class function is defined above in our function

# definitons. 

data[feature_name] = data.apply(lambda x: impute_age_by_class(x['Pclass'], 

                                                data,

                                                x), axis = 1)
# Now verifying that there are no more null values

data[feature_name].isnull().sum()
# Next we'll graph the distribution to check for skewness.

x = data[feature_name]

plt.hist(x, bins=25)

plt.xlabel('age')

plt.ylabel('Frequency')
new_feat = f'{feature_name}_scaled'

data[new_feat] = min_max(feature_name, data)

data.head()
# Next we'll make a box plot to get an idea of how significant the feature

# will be for predicting survival.

box_plot("Survived","Age_scaled", data)
# setting the feature name we're examining so that we don't have to re-type it

feature_name = "Fare"

# Checking for missing valeus

data[feature_name].isnull().sum()
data[data[feature_name].isnull()]
box_plot("Pclass","Fare", data)
# The impute_fare_by_class function is defined above in our function

# definitons. It also works the same as the impute_age_by_class function.

data[feature_name] = data.apply(lambda x: impute_fare_by_class(x['Pclass'], 

                                                data,

                                                x), axis = 1)
# Next we'll graph the distribution to check for skewness.

x = data[feature_name]

plt.hist(x, bins=25)

plt.xlabel('Fare')

plt.ylabel('Frequency')
log_fare = np.log(data[feature_name])

x = log_fare

plt.hist(x, bins=30)

plt.xlabel('log_Fare')

plt.ylabel('Frequency')
data['log_Fare'] = log_fare

new_feat = f'log_{feature_name}_scaled'

data[new_feat] = min_max('log_Fare', data)

data.head()
# Next we'll make a box plot to get an idea of how significant the feature

# will be for predicting survival.

box_plot("Survived","log_Fare_scaled", data)
# setting the feature name we're examining so that we don't have to re-type it

feature_name = "Cabin"

# checking for missing values

data[feature_name].isnull().sum()
# setting the feature name we're examining so that we don't have to re-type it

feature_name = "Embarked"

# checking for missing values

data[feature_name].isnull().sum()
data[data[feature_name].isnull()]
ax = sb.boxplot(x=data['Embarked'], y=data['Fare'])

ax.set_xticklabels(ax.get_xticklabels(),rotation=-90)

ax.set(ylim=(0, 100))
interaction_plot("Embarked", 'Pclass', data)
data[feature_name] = data[feature_name].fillna('C')
data[feature_name].isnull().sum()
# Next we'll make a box plot to get an idea of how significant the feature

# will be for predicting survival.

interaction_plot("Survived", 'Embarked', data)
# setting the feature name we're examining so that we don't have to re-type it

feature_name = "Sex"

# checking for missing values

data[feature_name].isnull().sum()
interaction_plot("Survived", 'Sex', data)
# setting the feature name we're examining so that we don't have to re-type it

feature_name = "SibSp"

# checking for missing values

data[feature_name].isnull().sum()
data[feature_name].describe()
# lets take a look at the graph of the distribution. 

x = data[feature_name]

plt.hist(x, bins=8)

plt.xlabel('SibSp')

plt.ylabel('Frequency')
# I'll scale the data now.

data[f'{feature_name}_scaled'] = min_max(feature_name, data)
data.head()
# Next we'll make a box plot to get an idea of how significant the feature

# will be for predicting survival.

box_plot("Survived","SibSp_scaled", data)
# setting the feature name we're examining so that we don't have to re-type it

feature_name = "Parch"

# checking for missing values

data[feature_name].isnull().sum()
data[feature_name].describe()
x = data[feature_name]

plt.hist(x, bins=9)

plt.xlabel('Parch')

plt.ylabel('Frequency')
# I'll scale the data now.

data[f'{feature_name}_scaled'] = min_max(feature_name, data)
data.head()
box_plot("Survived","Parch_scaled", data)
# dummies

X_dummies = pd.get_dummies(data[["Parch_scaled", "log_Fare_scaled", "FamMort", 

                                 "Title", "Pclass", "Embarked", "Sex"]])

# splitting

X_dummies_train = X_dummies.iloc[0:890]

X_dummies_test = X_dummies.iloc[891:]



# creating labels column

Y = data.iloc[0:890]["Survived"]

# creating dataframe to store predictions

test_frame = pd.DataFrame(data.iloc[891:]["PassengerId"])
# creating categorical frames

X_dummies_cat = X_dummies_train.drop(["Parch_scaled", "log_Fare_scaled"],

                                     axis = 1)

X_dummies_cat_t = X_dummies_test.drop(["Parch_scaled",  "log_Fare_scaled"], 

                                      axis = 1)

# creating numeric frames

X_num = X_dummies_train[["Parch_scaled", "log_Fare_scaled"]]

X_num_t = X_dummies_test[["Parch_scaled", "log_Fare_scaled"]]
# creating data frames to store our predicted probabilities in. We can add

# predicted probabilites to this as we model and then analyze Mean Squared

# Error at the end to evaluate them.

train_result_frame = pd.DataFrame(Y)

test_result_frame = pd.DataFrame()
# first step is setting up cross validation. The function train_test_split

# will randomly split our training set into training and a validation set. 

# validation sets are used to get an idea of how well the model will

# perform on unseen data.

x_train, x_valid, y_train, y_valid = train_test_split(

    X_dummies_train, Y, test_size=0.25, random_state=0)
# fitting

clf = tree.DecisionTreeClassifier().fit(x_train, y_train)

# checking accuracy on validation set

clf.score(x_valid, y_valid)
# fitting on entire training set

clf = tree.DecisionTreeClassifier().fit(X_dummies_train, Y)

# predicting on the test set

test_frame["Survived"] = clf.predict(X_dummies_test).astype('int32')

# writing to a file

test_frame.to_csv('/kaggle/working/decision_tree.csv', index = False)



# storing the predicted probabilities in dataframe for MSE analysis

train_result_frame['dt'] = [x[1] for x in clf.predict_proba(X_dummies_train)]

test_result_frame['dt'] = [x[1] for x in clf.predict_proba(X_dummies_test)]
# creating training and validation sets

x_train, x_valid, y_train, y_valid = train_test_split(

    X_dummies_train, Y, test_size=0.25, random_state=0)
# fitting 

logr = LogisticRegression(solver = 'lbfgs').fit(x_train, y_train)

# checking validation accuracy

logr.score(x_valid, y_valid)
# fitting to full training data

logr = LogisticRegression(solver = 'lbfgs').fit(X_dummies_train, Y)

# predicting the test set

test_frame["Survived"] = logr.predict(X_dummies_test).astype('int32')

# writing results to a file

test_frame.to_csv('/kaggle/working/logistic_reg.csv', index = False)



# storing the predicted probabilities in dataframe for MSE analysis

train_result_frame['logr'] = [x[1] for x in 

                                  logr.predict_proba(X_dummies_train)]

test_result_frame['logr'] = [x[1] for x in 

                                 logr.predict_proba(X_dummies_test)]
# the number of random samples and decision trees to build

n_estimators = [100, 200, 500, 600, 750, 1000, 1250]

# the min number of samples allowed at each decision tree node (controls

# against overfitting)

min_samples_split = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

# the min number of samples allowed at each decision tree leaf (controls

# against overfitting)

min_samples_leaf = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

# placing the parameter values into a dictionary

grid_param = {'n_estimators': n_estimators,

              'min_samples_split': min_samples_split,

              'min_samples_leaf': min_samples_leaf}



# initializing the model with a random state

random_forest = rfc(random_state=4538756)



# initializing the cross validator 

RFC_random = RandomizedSearchCV(estimator = random_forest, 

                             param_distributions = grid_param,

                             n_iter = 120,

                             verbose=2,

                             cv = 5,

                             random_state = 857436,

                             n_jobs = -1)

# starting the tuner

RFC_random.fit(X_dummies_train, Y)

print(RFC_random.best_params_)

print(f'score: {RFC_random.best_score_}')
# initializing the mdoel using best selected hyper-parameters

random_forest = rfc(n_estimators = 750, min_samples_split = 10, 

                    min_samples_leaf = 5, random_state = 27394652)

# fitting to entire training set

randf = random_forest.fit(X_dummies_train, Y)

# predicting the test set

test_frame["Survived"] = randf.predict(X_dummies_test).astype('int32')

# writing to a file

test_frame.to_csv('/kaggle/working/random_forest.csv', index = False)



# storing the predicted probabilities in dataframe for MSE analysis

train_result_frame['randf'] = [x[1] for x in 

                                  randf.predict_proba(X_dummies_train)]

test_result_frame['randf'] = [x[1] for x in 

                                 randf.predict_proba(X_dummies_test)]
# the number of trees to build

n_estimators = [100, 250, 500, 750, 1000, 1250]

# the learning rate to apply with every tree built

learning_rate = [0.01, 0.05, 0.1, 0.2, 0.3]

# the min number of samples allowed at each decision tree node (controls

# against overfitting)

min_samples_split = [20, 25, 30, 35, 40, 45, 50, 55]

# the min number of samples allowed at each decision tree leaf (controls

# against overfitting)

min_samples_leaf = [20, 25, 30, 35, 40, 45, 50, 55]

# placing the parameter values into a dictionary

grid_param = {'n_estimators': n_estimators,

              'learning_rate': learning_rate,

              'min_samples_split': min_samples_split,

              'min_samples_leaf': min_samples_leaf}



# initialize model with a random state

gradient_boosted_trees = gbc(random_state=24352345)

# initialize the cross validator

GB_random = RandomizedSearchCV(estimator = gradient_boosted_trees, 

                             param_distributions = grid_param,

                             n_iter = 120,

                             verbose=2,

                             cv = 5,

                             random_state = 47567,

                             n_jobs = -1)

# starting the tuner

GB_random.fit(X_dummies_train, Y)

# print result

print(GB_random.best_params_)

print(f'score: {GB_random.best_score_}')
# initializing the mdoel using best selected hyper-parameters

grad_booster = gbc(n_estimators = 1000, min_samples_split = 45, 

                    min_samples_leaf = 20, learning_rate = 0.05, 

                   random_state = 34737)

# fitting to entire training set

gradb = grad_booster.fit(X_dummies_train, Y)

# predicting the test set

test_frame["Survived"] = gradb.predict(X_dummies_test).astype('int32')

# writing to a file

test_frame.to_csv('/kaggle/working/gradient_boosted_trees.csv', index = False)



# storing the predicted probabilities in dataframe for MSE analysis

train_result_frame['gbc'] = [x[1] for x in 

                                  gradb.predict_proba(X_dummies_train)]

test_result_frame['gbc'] = [x[1] for x in 

                                 gradb.predict_proba(X_dummies_test)]
# define the values of k to test

k = [5, 11, 15, 21, 25, 31, 35, 41, 45, 51, 55, 61, 65, 71, 75, 81, 85, 91]

# placing the parameter values into a dictionary

grid_param = {'n_neighbors': k}



# initialize model with given distance metric

model = knn(metric = 'jaccard')

# initialize the cross validator

KNN_random = GridSearchCV(estimator = model, 

                             param_grid = grid_param,

                             verbose = 2,

                             cv = 5,

                             n_jobs = -1)

# begin tuning

KNN_random.fit(X_dummies_cat, Y)

# print results

print(KNN_random.best_params_)

print(f'score: {KNN_random.best_score_}')
# initializing the mdoel using best selected hyper-parameters

knn_grad = knn(metric = 'jaccard', n_neighbors = 11)

# fitting to entire training set

knn_c = knn_grad.fit(X_dummies_cat, Y)

# predicting the test set

test_frame["Survived"] = knn_c.predict(X_dummies_cat_t).astype('int32')

# writing to a file

test_frame.to_csv('/kaggle/working/knn_categorical.csv', index = False)



# storing the predicted probabilities in dataframe for MSE analysis

train_result_frame['knn_cat'] = [x[1] for x in 

                                  knn_c.predict_proba(X_dummies_cat)]

test_result_frame['knn_cat'] = [x[1] for x in 

                                 knn_c.predict_proba(X_dummies_cat_t)]
# define the values of k to test

k = [5, 11, 15, 21, 25, 31, 35, 41, 45, 51, 55, 61, 65, 71, 75, 81, 85, 91]

# placing the parameter values into a dictionary

grid_param = {'n_neighbors': k}



# initialize model with given distance metric

model = knn(metric = 'euclidean')

# initialize the cross validator

KNN_random = GridSearchCV(estimator = model, 

                             param_grid = grid_param,

                             verbose = 2,

                             cv = 5,

                             n_jobs = -1)

# begin tuning

KNN_random.fit(X_num, Y)

# print results

print(KNN_random.best_params_)

print(f'score: {KNN_random.best_score_}')
# initializing the mdoel using best selected hyper-parameters

knn_grad = knn(metric = 'euclidean', n_neighbors = 61)

# fitting to entire training set

knn_c = knn_grad.fit(X_num, Y)

# predicting the test set

test_frame["Survived"] = knn_c.predict(X_num_t).astype('int32')

# writing to a file

test_frame.to_csv('/kaggle/working/knn_numeric.csv', index = False)



# storing the predicted probabilities in dataframe for MSE analysis

train_result_frame['knn_num'] = [x[1] for x in 

                                  knn_c.predict_proba(X_num)]

test_result_frame['knn_num'] = [x[1] for x in 

                                 knn_c.predict_proba(X_num_t)]
# fitting K-NN numeric with the best selected hyperparameters

knn_num = knn(metric = 'euclidean', n_neighbors = 61)

knn_num.fit(X_num, Y)

# fitting K-NN categorical with the best selected hyperparameters

knn_cat = knn(metric = 'jaccard', n_neighbors = 11)

knn_cat.fit(X_dummies_cat, Y)



# creating a data frame to store the probablities in

frame = pd.DataFrame(Y)

# saving calculated probabilities for the numeric training data

frame["num_prob"] = [x[1] for x in knn_num.predict_proba(X_num)]

# saving calculated probabilities for the categorical training data

frame["cat_prob"] = [x[1] for x in knn_cat.predict_proba(X_dummies_cat)]



# define the values of k to test

k = [5, 11, 15, 21, 25, 31, 35, 41, 45, 51, 55, 61, 65, 71, 75, 81, 85, 91]

# placing the parameter values into a dictionary

grid_param = {'n_neighbors': k}



# initialize model with given distance metric

knn_full = knn(metric = 'euclidean')

# initialize the cross validator

KNN_random = GridSearchCV(estimator = knn_full, 

                             param_grid = grid_param,

                             verbose = 2,

                             cv = 5,

                             n_jobs = -1)

# begin tuning

KNN_random.fit(frame[["num_prob", "cat_prob"]], frame["Survived"])

# print results

print(KNN_random.best_params_)

print(f'score: {KNN_random.best_score_}')
# fitting K-NN numeric with the best selected hyperparameters

knn_num = knn(metric = 'euclidean', n_neighbors = 61)

knn_num.fit(X_num, Y)

# fitting K-NN categorical with the best selected hyperparameters

knn_cat = knn(metric = 'jaccard', n_neighbors = 11)

knn_cat.fit(X_dummies_cat, Y)



# creating a data frame to store the training probablities in

frame = pd.DataFrame(Y)

# saving calculated probabilities for the numeric training data

frame["num_prob"] = [x[1] for x in knn_num.predict_proba(X_num)]

# saving calculated probabilities for the categorical training data

frame["cat_prob"] = [x[1] for x in knn_cat.predict_proba(X_dummies_cat)]



# creating a data frame to store the test probablities in

frame_test = pd.DataFrame()

# saving calculated probabilities for the numeric test data

frame_test["num_prob"] = [x[1] for x in knn_num.predict_proba(X_num_t)]

# saving calculated probabilities for the categorical test data

frame_test["cat_prob"] = [x[1] for x in knn_cat.predict_proba(X_dummies_cat_t)]



# fitting full K-NN with the best selected hyperparameters

knn_full = knn(metric = 'euclidean', n_neighbors = 31)

knn_full = knn_full.fit(frame[["num_prob", "cat_prob"]], frame["Survived"])



# predicting on the test set probabilities

test_frame["Survived"] = knn_full.predict(

    frame_test[["num_prob", "cat_prob"]]

).astype('int32')

# saving the predictions

test_frame.to_csv('/kaggle/working/knn_full.csv', index = False)



# storing the predicted probabilities in dataframe for MSE analysis

train_result_frame['knn'] = [x[1] for x in 

                                  knn_full.predict_proba(

                                      frame[["num_prob", "cat_prob"]])]

test_result_frame['knn'] = [x[1] for x in 

                                 knn_full.predict_proba(

                                     frame_test[["num_prob", "cat_prob"]])]
def create_neural_net(in_shape, lyrs=[4], act='relu', opt='Adam', dr=0.0):

    # set random seed for reproducibility

    seed(37556)

    tf.random.set_seed(37556)

    

    # initialize the model object

    model = Sequential()

    

    # create first hidden layer

    model.add(Dense(lyrs[0], input_dim=in_shape, activation=act))

    

    # create additional hidden layers

    for i in range(1,len(lyrs)):

        model.add(Dense(lyrs[i], activation=act))

    

    # add dropout, default is none

    model.add(Dropout(dr))

    

    # create output layer

    model.add(Dense(1, activation='sigmoid'))  # output layer

    

    model.compile(loss='binary_crossentropy', optimizer=opt,

                  metrics=['accuracy'])

    

    return model
# creating a neural network

single_net = create_neural_net(X_dummies_train.shape[1], lyrs =[4])

single_net.summary()
# fitting the network to the training data with arbitrary hyper-parameters.

# this is just an example to show off some cool things keras can do.

training = single_net.fit(X_dummies_train, Y, epochs=100, batch_size=32,

                         validation_split=0.25, verbose=0)

# outputting the validation accuracy

val_acc = np.mean(training.history['val_accuracy'])

print("\n%s: %.2f%%" % ('val_acc', val_acc*100))
# plotting the training history

plt.plot(training.history['accuracy'])

plt.plot(training.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
# We need this specific version of scikit-learn for hyperparameter tuning 

!pip install scikit-learn==0.21.2
# wrap the nerual network in an sklearn class so we can plug it into 

# RandomizedSearchCV

neural_net = KerasClassifier(build_fn=create_neural_net, 

                             in_shape = X_dummies_train.shape[1], verbose = 0)

# defining the batch size

batch_size = [1, 16, 32, 64]

# defining the number of epochs

epochs = [25, 50, 100, 150]

# defining the size and number of layers

layers = [[4], [8], [12], [4, 4], [8, 4], [8, 8], [12, 8], [12, 4], [12, 8, 4]]

# defining the rate of the dropout layer

drops = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]



# initialize model with given hyper-parameters

grid_param = {'batch_size': batch_size,

             'epochs': epochs,

             'dr': drops,

            'lyrs': layers}

# initialize the cross validator

nn_random = RandomizedSearchCV(estimator = neural_net, 

                             param_distributions = grid_param,

                             n_iter = 200,

                             verbose=2,

                             cv = 5,

                             random_state = 456745,

                             n_jobs = -1)

# begin tuning

nn_random.fit(X_dummies_train, Y)



# print results

print(nn_random.best_params_)

print(f'score: {nn_random.best_score_}')
# initializing the mdoel using best selected hyper-parameters

nn = KerasClassifier(build_fn=create_neural_net, 

                        in_shape = X_dummies_train.shape[1],

                        lyrs=[12, 8, 4], epochs=50, dr=0.1, batch_size=1, 

                         verbose=0)

# fitting to entire training set

nn.fit(X_dummies_train, Y)

# predicting the entire test set

test_frame["Survived"] = nn.predict(X_dummies_test).astype('int32')

# writing to a file

test_frame.to_csv('/kaggle/working/nn.csv', index = False)



# storing the predicted probabilities in dataframe for MSE analysis

train_result_frame['nn'] = [x[1] for x in 

                                  nn.predict_proba(X_dummies_train)]

test_result_frame['nn'] = [x[1] for x in 

                                 nn.predict_proba(X_dummies_test)]
test_result_frame.to_csv('/kaggle/working/test_results.csv', index = False)

#test_result_frame = pd.read_csv('./result_data/test_results.csv')



train_result_frame.to_csv('/kaggle/working/train_results.csv', index = False)

#train_result_frame = pd.read_csv('./result_data/train_results.csv')

train_result_frame.head()
mse = {col: 0 for col in train_result_frame.columns if col != "Survived"}

for col in mse:

    error = mse[col]

    for _,row in train_result_frame[['Survived', col]].iterrows():

        error += math.pow(row['Survived'] - row[col], 2)

    error /= train_result_frame.shape[0]

    mse[col] = error
for k, v in sorted(mse.items(), key=lambda item: item[1]):

    print(f'{k:10} MSE: {mse[k]:.3f}')