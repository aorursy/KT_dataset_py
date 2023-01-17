# Import the libraries

import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier 

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report,accuracy_score

from sklearn.neighbors import KNeighborsClassifier





# Read the dataset

data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.info()
#Removing the time column since it would not help in model training

data.drop(["Time"],axis=1,inplace = True)
# Lets check the missing values in case any

data.isnull().sum()
# Correlation of variables with the target class.

corr = data.corr()

corr["Class"]
# Removing the variables which can affect the output stream

#These are features either are too negatively correlated with the target or too much positive with target(may surpress other features)

COL=["V3","V10","V12","V14","V16","V17","V18"]

data.drop(COL,axis=1,inplace=True)
# Lets normalise the amount column

# Need not perform scaling of other features since all values have been scaled because of the PCA applied before the data was released

from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

data = data.drop(['Amount'],axis=1)
#Checking the ratio of classes with respect to each other 

data["Class"].value_counts()
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.



# Lets shuffle the data before creating the subsamples



df = data.sample(frac=1)



# NO of fraud instances :: 492 rows.

fraud_df = df.loc[df['Class'] == 1]

non_fraud_df = df.loc[df['Class'] == 0][:492]



normal_distributed_df = pd.concat([fraud_df, non_fraud_df])



# Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)



new_df.head()
# Lets split the data into features and target

features = new_df.drop(["Class"],axis =1)

target = new_df["Class"]

from sklearn.model_selection import train_test_split



# Splitting the data into train and test

X_train,X_test, Y_train,Y_test = train_test_split(features, target, test_size = .3, random_state =5)
# Logistic Regression 

# Solver is liblinear since its supports both l1 and l2 penalities

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],"solver" :['liblinear']}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, Y_train)

# We automatically get the logistic regression with the best parameters.



print(grid_log_reg.best_estimator_)



knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(X_train, Y_train)

# KNears best estimator

print(grid_knears.best_estimator_)



# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train, Y_train)



# SVC best estimator



print(grid_svc.best_estimator_)



# DecisionTree Classifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(X_train, Y_train)



# tree best estimator

print(grid_tree.best_estimator_)


k = KFold(n_splits = 5)

classifiers = {

    'Logistic Regression': LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,

                                               intercept_scaling=1, l1_ratio=None, max_iter=100,

                                               multi_class='auto', n_jobs=None, penalty='l1',

                                               random_state=None, solver='liblinear', tol=0.0001, verbose=0,

                                               warm_start=False),

               'Support Vector Machine' : SVC(C=0.9, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,

                                                decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',

                                                max_iter=-1, probability=False, random_state=None, shrinking=True,

                                                tol=0.001, verbose=False),

               'Random Forest Classifier': RandomForestClassifier(),

               'Decision Tree Algorithm':DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',

                                                                   max_depth=2, max_features=None, max_leaf_nodes=None,

                                                                   min_impurity_decrease=0.0, min_impurity_split=None,

                                                                   min_samples_leaf=5, min_samples_split=2,

                                                                   min_weight_fraction_leaf=0.0, presort='deprecated',

                                                                   random_state=None, splitter='best'),

               'KNN':KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                                             metric_params=None, n_jobs=None, n_neighbors=3, p=2,

                                             weights='uniform')

              }

def model(m):

    for i in m:

        print(i)

        print('-'*100)

        m[i].fit(X_train,Y_train.values.ravel())

        prediction = m[i].predict(X_test)

        print('Classification Report')        

        cr = classification_report(Y_test,prediction,output_dict=True)

        print(pd.DataFrame(cr).transpose())

        print('='*100)

        accuracy = accuracy_score(Y_test.values.ravel(),prediction)

        print('Accuracy Score :  ',accuracy)

        print('='*100)

        print('Cross Validation Score')

        cv=cross_val_score(m[i],X_train,Y_train,cv=k,scoring='accuracy')

        print(cv.mean())

        print('-'*100)
model(classifiers)
from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

import numpy as np

import matplotlib.pyplot as plt

def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20,14), sharey=True)

    if ylim is not None:

        plt.ylim(*ylim)

    # First Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)

    ax1.set_xlabel('Training size (m)')

    ax1.set_ylabel('Score')

    ax1.grid(True)

    ax1.legend(loc="best")

    

    # Second Estimator 

    train_sizes, train_scores, test_scores = learning_curve(

        estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax2.set_title("Knears Neighbors Learning Curve", fontsize=14)

    ax2.set_xlabel('Training size (m)')

    ax2.set_ylabel('Score')

    ax2.grid(True)

    ax2.legend(loc="best")

    

    # Third Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax3.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax3.set_title("Support Vector Classifier \n Learning Curve", fontsize=14)

    ax3.set_xlabel('Training size (m)')

    ax3.set_ylabel('Score')

    ax3.grid(True)

    ax3.legend(loc="best")

    

    # Fourth Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax4.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax4.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax4.set_title("Decision Tree Classifier \n Learning Curve", fontsize=14)

    ax4.set_xlabel('Training size (m)')

    ax4.set_ylabel('Score')

    ax4.grid(True)

    ax4.legend(loc="best")

    return plt
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plot_learning_curve(classifiers["Logistic Regression"],classifiers["KNN"], classifiers["Support Vector Machine"], classifiers["Decision Tree Algorithm"], X_train, Y_train, (0.87, 1.01), cv=cv, n_jobs=4)
from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_predict

# Create a DataFrame with all the scores and the classifiers names.



log_reg_pred = cross_val_predict(classifiers["Logistic Regression"], X_train, Y_train, cv=5,

                             method="decision_function")



knears_pred = cross_val_predict(classifiers["KNN"], X_train, Y_train, cv=5)



svc_pred = cross_val_predict(classifiers["Support Vector Machine"], X_train, Y_train, cv=5,

                             method="decision_function")



tree_pred = cross_val_predict(classifiers["Decision Tree Algorithm"], X_train, Y_train, cv=5)

from sklearn.metrics import roc_auc_score



print('Logistic Regression: ', roc_auc_score(Y_train, log_reg_pred))

print('KNears Neighbors: ', roc_auc_score(Y_train, knears_pred))

print('Support Vector Classifier: ', roc_auc_score(Y_train, svc_pred))

print('Decision Tree Classifier: ', roc_auc_score(Y_train, tree_pred))
import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Activation

from keras.layers.core import Dense

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy



#Size of the input of the input layer

n_inputs = X_train.shape[1]



undersample_model = Sequential([

    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),

    Dense(32, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)),

    Dense(2, activation='softmax') #2 neurons because we have 2 output classes 

])
undersample_model.summary()
keras.utils.plot_model(undersample_model)
#using sparse_categorial because there is a lot of sparse of values in the target so it is preferred to use this loss

# using accuracy since it is classification problem 

undersample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#Spliting into train, Dev set and test set

# Train = 60% Valid = 30% Test = 1% on original dataset 

X= data.drop(["Class"],axis = 1)

Y = data["Class"]

org_x_train_f,org_x_test,org_y_train_f,org_y_test = train_test_split(X,Y,test_size = .1,random_state = 1)

org_x_train,org_x_valid,org_y_train,org_y_valid = train_test_split(org_x_train_f,org_y_train_f,test_size = .3,random_state = 1)
# Using random shuffling in the dataset while fitting

history=undersample_model.fit(X_train, Y_train, batch_size=64, epochs=20, shuffle=True, verbose=2,validation_data = (org_x_valid,org_y_valid))
#Lets check out the weights and biases in each layer

for i in range(3):

    print('Layer',i+1)

    print("-"*10)

    hidden_layer  = undersample_model.layers[i]

    weights,biases = hidden_layer.get_weights()

    print('Weights')

    print(weights)

    print("Weight shape",weights.shape)

    print("*"*10)

    print('Bias')

    print(biases)

    print("Bias shape",biases.shape)

    print("="*10)


pd.DataFrame(history.history).plot(figsize=(10,10))

plt.grid(True)

plt.gca().set_ylim(0,1)

plt.show()
# Evaluation on test test 

acc=undersample_model.evaluate(org_x_test,org_y_test)

print('Loss : {} ,Accuracy : {}'.format(acc[0],acc[1]*100))