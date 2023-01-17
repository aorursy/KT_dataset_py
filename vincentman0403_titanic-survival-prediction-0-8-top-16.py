import pandas as pd
import numpy as np
from collections import Counter
import math
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
# turn off warning: SettingWithCopyWarning
pd.set_option('chained_assignment', None)
df_csv_train = pd.read_csv("../input/train.csv") # 891 samples
df_csv_train.head()
df_csv_test = pd.read_csv("../input/test.csv")  # 418 samples
df_csv_test.head()
dataset = pd.concat(objs=[df_csv_train, df_csv_test], axis=0, sort=True).reset_index(drop=True)
dataset.head(5)
# Fill Fare missing values with the median value
print('Null count of Fare before fillna: ', dataset["Fare"].isnull().sum())
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
print('Null count of Fare after fillna: ', dataset["Fare"].isnull().sum())
# Apply log to Fare to reduce skewness distribution
print('Skewness of Fare before log: ', stats.skew(dataset["Fare"]))
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
print('Skewness of Fare after log: ', stats.skew(dataset["Fare"]))
# Fill Embarked null values of dataset set with 'S' most frequent value
print('Null count of Embarked before fillna: ', dataset["Embarked"].isnull().sum())
dataset["Embarked"] = dataset["Embarked"].fillna("S")
print('Null count of Embarked after fillna: ', dataset["Embarked"].isnull().sum())
# convert Sex into categorical value 0 for male and 1 for female
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1})
print('NaN value count of Age before fillna: ', dataset["Age"].isnull().sum())
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)
for i in index_NaN_age:
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][(
            (dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (
            dataset['Parch'] == dataset.iloc[i]["Parch"]) & (
                    dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred):
        dataset['Age'].iloc[i] = age_pred
    else:
        dataset['Age'].iloc[i] = age_med
print('NaN value count of Age after fillna: ', dataset["Age"].isnull().sum())
# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
# Convert to categorical values Title
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master": 0, "Miss": 1, "Ms": 1, "Mme": 1, "Mlle": 1, "Mrs": 1, "Mr": 2, "Rare": 3})
dataset["Title"] = dataset["Title"].astype(int)
# Drop Name variable
dataset.drop(labels=["Name"], axis=1, inplace=True)
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
# Create new features for family size
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if s == 2 else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])
Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])  # Take prefix
    else:
        Ticket.append("X")
dataset["Ticket"] = Ticket
dataset = pd.get_dummies(dataset, columns=["Title"])
dataset = pd.get_dummies(dataset, columns=["Embarked"], prefix="Em")
dataset = pd.get_dummies(dataset, columns=["Cabin"], prefix="Cabin")
dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="T")
dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")
# Drop useless variables
dataset.drop(labels=["PassengerId"], axis=1, inplace=True)
df_test = dataset[dataset['Survived'].isnull()]
print('test data shape: ', df_test.shape)
df_train = dataset[dataset['Survived'].notnull()]
print('train data shape: ', df_train.shape)
def drop_outliers(df, n, features):
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    print('drop outlier samples id: ', multiple_outliers)

    df = df.drop(multiple_outliers, axis=0)
    
    return df
df_train = drop_outliers(df_train, 2, ["Age", "SibSp", "Parch", "Fare"])
print('After drop outlier samples, train data shape: ', df_train.shape)
# Convert Survived dtype as int
df_train['Survived'] = df_train['Survived'].astype(int)
# Split train data into x(features) and y(labels)
y_train = df_train.Survived
x_train = df_train.drop(['Survived'], axis=1)
# Standardize train data
x_train_std = StandardScaler().fit_transform(x_train.values)
y_train = y_train.values
clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, bootstrap=False)
kfold = StratifiedKFold(n_splits=10)
param_max_depth = [None]
param_min_samples_split = [2, 3, 10]
param_min_samples_leaf = [1, 3, 10]
param_max_features = [1, 3, 10]
param_n_estimators = [1100]
param_grid = {"max_depth": param_max_depth,
              "max_features": param_max_features,
              "min_samples_split": param_min_samples_split,
              "min_samples_leaf": param_min_samples_leaf,
              "n_estimators": param_n_estimators,
              }
gs = GridSearchCV(estimator=clf,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=kfold, iid=False)
start = time.time()
gs.fit(x_train, y_train)
end = time.time()
elapsed_train_time = 'Random Forest, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                                    int((end - start) % 60))
print(elapsed_train_time)
print('--------------------------------------------')
print(gs.best_estimator_)
print('--------------------------------------------')
print('Random Forest, train best score: {}'.format(gs.best_score_))
print('Random Forest, train best param: {}'.format(gs.best_params_))
random_forest_clf = gs.best_estimator_
param_C = [1, 10, 50, 100, 200, 300, 1000]
param_gamma = [0.0001, 0.001, 0.01, 0.1, 1.0]
param_grid = {'C': param_C, 'gamma': param_gamma, 'kernel': ['rbf']}
svm = SVC(random_state=0, verbose=False)
kfold = StratifiedKFold(n_splits=10)
gs = GridSearchCV(estimator=svm,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=kfold, iid=False)
start = time.time()
gs.fit(x_train_std, y_train)
end = time.time()
elapsed_train_time = 'SVM, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                          int((end - start) % 60))
print(elapsed_train_time)
print('--------------------------------------------')
print(gs.best_estimator_)
print('--------------------------------------------')
print('SVM, train best score: {}'.format(gs.best_score_))
print('SVM, train best param: {}'.format(gs.best_params_))
svm_clf = gs.best_estimator_
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_l1_ratio = np.arange(0.1, 1, 0.1)
param_grid = {'loss': ['hinge'], 'alpha': param_range, 'l1_ratio': param_l1_ratio}
kfold = StratifiedKFold(n_splits=5)
sgd = SGDClassifier(loss='hinge', verbose=0, max_iter=None, penalty='elasticnet', tol=1e-3)
gs = GridSearchCV(estimator=sgd,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=kfold, iid=False)
start = time.time()
gs.fit(x_train_std, y_train)
end = time.time()
elapsed_train_time = 'SGD with SVM, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                                   int((end - start) % 60))
print(elapsed_train_time)
print('--------------------------------------------')
print(gs.best_estimator_)
print('--------------------------------------------')
print('SGD with SVM at GridSearch, train best score: {}'.format(gs.best_score_))
print('SGD with SVM at GridSearch, train best param: {}'.format(gs.best_params_))
sgd_clf = gs.best_estimator_
model = Sequential()
model.add(Dense(units=40, input_dim=x_train.shape[1], kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(units=30, kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(units=30, kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(units=1, activation='sigmoid'))
print(model.summary())
adam = Adam(lr=0.01, decay=0.001, beta_1=0.9, beta_2=0.9)
model.compile(loss='binary_crossentropy',
              optimizer=adam, metrics=['accuracy'])
def show_train_history(train_history, train_acc, validation_acc, ylabel):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[validation_acc])
    epoch_num = len(train_history.epoch)
    final_epoch_train_acc = train_history.history[train_acc][epoch_num - 1]
    final_epoch_validation_acc = train_history.history[validation_acc][epoch_num - 1]
    plt.text(epoch_num, final_epoch_train_acc, 'train = {:.3f}'.format(final_epoch_train_acc))
    plt.text(epoch_num, final_epoch_validation_acc-0.01, 'valid = {:.3f}'.format(final_epoch_validation_acc))
    plt.title('Train History')
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.xlim(xmax=epoch_num+1)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    return final_epoch_train_acc, final_epoch_validation_acc
start = time.time()
train_history = model.fit(x=x_train_std,
                          y=y_train,
                          validation_split=0.1,
                          epochs=30,
                          shuffle=True,
                          batch_size=20, verbose=0)
end = time.time()
train_acc, validation_acc = show_train_history(train_history, 'acc', 'val_acc', 'accuracy')
train_loss, validation_loss = show_train_history(train_history, 'loss', 'val_loss', 'loss')
print('elapsed training time: {} min, {} sec '.format(int((end - start) / 60), int((end - start) % 60)))
print('train accuracy = {}, validation accuracy = {}'.format(train_acc, validation_acc))
print('train loss = {}, validation loss = {}'.format(train_loss, validation_loss))
# drop Survived of test data
x_test = df_test.drop('Survived', axis=1)
# standardize test data
x_test_std = StandardScaler().fit_transform(x_test.values)
pd.DataFrame({"PassengerId": np.arange(892, 1310), "Survived": model.predict_classes(x_test_std).ravel()}).to_csv(
    'submission_mlp.csv', header=True, index=False)
pd.DataFrame({"PassengerId": np.arange(892, 1310), "Survived": random_forest_clf.predict(x_test).astype(int)}).to_csv(
    'submission_rf.csv', header=True, index=False)
pd.DataFrame({"PassengerId": np.arange(892, 1310), "Survived": svm_clf.predict(x_test_std).astype(int)}).to_csv(
    'submission_svm.csv', header=True, index=False)
pd.DataFrame({"PassengerId": np.arange(892, 1310), "Survived": sgd_clf.predict(x_test_std).astype(int)}).to_csv(
    'submission_sgd.csv', header=True, index=False)
