# import useful libraries

import numpy as np #

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from time import time

from sklearn.metrics import roc_auc_score





# Import the supervised learning models from sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression



#disable warnings

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv("../input/census.csv")

data_test = pd.read_csv("../input/test_census.csv").drop('Unnamed: 0',1)

#make a copy of data_train to overwrite during feature engineering

train = data_train[:]
# train and test set shape

print("Train set shape:", train.shape)

print("Test set shape:", data_test.shape)
# first glance on train set

train.head()
#first glance on test data

data_test.head()
# Check info for train and test dataset

data_train.info()

print("----------------------------------")

data_test.info()
# Total number of records

n_records = train.shape[0]



# Print the results

print("Total number of records: {}".format(n_records))

#explore unique values for income, to use it below 

data_train.income.unique()
# Number of records where individual's income is more than $50,000

n_greater_50k = train[train.income == '>50K' ].shape[0]



# Number of records where individual's income is at most $50,000

n_at_most_50k = train[train.income == '<=50K' ].shape[0]



# Percentage of individuals whose income is more than $50,000

greater_percent = round((n_greater_50k/n_records)*100 ,2)



# Print the results

print("Individuals making more than $50,000: {}".format(n_greater_50k))

print("Individuals making at most $50,000: {}".format(n_at_most_50k))

print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))
# transform target into new variable called income:

income=data_train.income.map({'<=50K': 0, '>50K':1})

income.head()
# check how many unique values each feature has:

for column in data_train.columns:

    print(column, len(train[column].unique()))
categorical = ['workclass', 'education_level', 'marital-status', 'occupation', 'relationship', 

               'race', 'sex', 'native-country']

continues = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'education-num']
# for each categorical features pring unique values:

for column in categorical:

    print(column, train[column].unique())
# plot disctribution and check skewness:

for column in continues:

    a = sns.FacetGrid(train, aspect=4 )

    a.map(sns.kdeplot, column, shade= True )

    a.add_legend()

    print('Skew for ',str(column), train[column].skew())
skewed = ['capital-gain', 'capital-loss']

# Log-transform the skewed features (create function to use later for test set)

def log_transform(data):

    return data[skewed].apply(lambda x: np.log(x + 1))

    

train[skewed] = log_transform(train)



# Visualize the new log distributions

for column in skewed:

    a = sns.FacetGrid(train, aspect=4 )

    a.map(sns.kdeplot, column, shade= True )

    a.add_legend()

    print('Skew for ',str(column), train[column].skew())
for column in continues:

    sns.boxplot(train[column])

    plt.show()
from sklearn.preprocessing import MinMaxScaler

#normalizing numerical features. Create function to use later on test data



def normalize(data):

    

    scaler = MinMaxScaler()

    data=scaler.fit_transform(data[continues])

    return data



train[continues]= normalize(train)

train.head(100)
# One-hot encode thedata using pandas.get_dummies()

features_final = pd.get_dummies(train.drop(['income'],1))



# Print the number of features after one-hot encoding

encoded = list(features_final.columns)

print("{} total features after one-hot encoding.".format(len(encoded)))

# check correlation between features: 

data = pd.concat([features_final, income], axis =1)

plt.figure(figsize=(30,28))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data.corr(),linewidths=0.1,vmax=1.0, 

            square=True,linecolor='white')
# Import train_test_split

from sklearn.model_selection import train_test_split



# Split the 'features' and 'income' data into training and testing sets

X_train, X_val, y_train, y_val = train_test_split(features_final, 

                                                    income, 

                                                    test_size = 0.2, 

                                                    random_state = 0)



# Show the results of the split

print("Training set has {} samples.".format(X_train.shape[0]))

print("Testing set has {} samples.".format(X_val.shape[0]))
def evaluate(results):

    """

    Visualization code to display results of various learners.

    

    inputs:

      - learners: a list of supervised learners

      - stats: a list of dictionaries of the statistic results from 'train_predict()'

      - accuracy: The score for the naive predictor

      - f1: The score for the naive predictor

    """

  

    # Create figure

    fig, ax = plt.subplots(2, 2, figsize = (18,10))



    # Constants

    bar_width = 1

    colors = ['r','g','b','c', 'm', 'y']

    

    # Super loop to plot four panels of data

    for k, learner in enumerate(results.keys()):

        for j, metric in enumerate(['train_time', 'roc_train',  'pred_time', 'roc_test']):



                ax[j//2, j%2].bar(k*bar_width, results[learner][metric], width = bar_width, color = colors[k])

    

    # Add unique y-labels

    ax[0, 0].set_ylabel("Time (in seconds)")

    ax[0, 1].set_ylabel("ROC-AUC Score")

    ax[1, 0].set_ylabel("Time (in seconds)")

    ax[1, 1].set_ylabel("ROC-AUC Score")

    

    # Add titles

    ax[0, 0].set_title("Model Training")

    ax[0, 1].set_title("ROC-AUC Score on Training Subset")

    ax[1, 0].set_title("Model Predicting")

    ax[1, 1].set_title("ROC-AUC Score on Testing Set")

       

    # Set y-limits for score panels

    ax[0, 1].set_ylim((0, 1))

    ax[1, 1].set_ylim((0, 1))



    # Create patches for the legend

    patches = []

    for i, learner in enumerate(results.keys()):

        patches.append(mpatches.Patch(color = colors[i], label = learner))

    ax[1, 0].legend(handles = patches)

    

    # Aesthetics

    plt.suptitle("Performance Metrics for Supervised Learning Models", fontsize = 16, y = 1.10)

    plt.tight_layout()

    plt.show()
# Display inline matplotlib plots with IPython

from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')





def train_predict(learner, X_train, y_train, X_test, y_test): 

    '''

    inputs:

       - learner: the learning algorithm to be trained and predicted on

       - sample_size: the size of samples (number) to be drawn from training set

       - X_train: features training set

       - y_train: income training set

       - X_val: features testing set

       - y_val: income testing set

    '''

    

    results = {}

    

    # Fit the learner to the training data 

    start = time() # Get start time

    learner.fit(X_train, y_train)

    end = time() # Get end time

    

    # Calculate the training time

    results['train_time'] = end - start

        

    # Get the predictions on the test set(X_test),

    #       then get predictions on the first 300 training samples(X_train) using .predict()

    start = time() # Get start time

    predictions_test = learner.predict(X_val)

    predictions_train = learner.predict(X_train[:300])

    end = time() # Get end time

    

    # Calculate the total prediction time

    results['pred_time'] = end - start

            

    # Compute accuracy on the first 300 training samples which is y_train[:300]

    results['roc_train'] = roc_auc_score(y_train[:300], predictions_train)

        

    # Compute accuracy on test set using accuracy_score()

    results['roc_test'] = roc_auc_score(y_val, predictions_test)

              

    # Return the results

    return results
random_state =42

n_estimators =100



# Initialize the three models

clf_A = GaussianNB()

clf_B = KNeighborsClassifier()

clf_C = LogisticRegression(random_state= random_state)

clf_D = RandomForestClassifier(random_state= random_state, n_estimators = n_estimators)

clf_E = GradientBoostingClassifier(n_estimators = n_estimators, random_state = random_state)

clf_F = AdaBoostClassifier(n_estimators = n_estimators, random_state = random_state)



# Collect results on the learners

results = {}

for clf in [clf_A, clf_B, clf_C, clf_D, clf_E, clf_F]:

    clf_name = clf.__class__.__name__

    results[clf_name] = {}

    results[clf_name] = train_predict(clf, X_train, y_train, X_val, y_val)
# Run metrics visualization for the three supervised learning models chosen

evaluate(results)
# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries

from sklearn.metrics import make_scorer 

from sklearn.model_selection import GridSearchCV
clf = AdaBoostClassifier(random_state = random_state)



# Create the parameters list you wish to tune, using a dictionary if needed.

parameters = {'n_estimators': range(20,1021,100)}



#Make an fbeta_score scoring object using make_scorer()

scorer = make_scorer(roc_auc_score)



# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = GridSearchCV(clf, param_grid = parameters, scoring= scorer,  verbose=1, n_jobs =-1)



#  Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train, y_train)



# Get the estimator

best_clf = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train, y_train)).predict(X_val)

best_predictions_val = best_clf.predict(X_val)

best_predictions_train = best_clf.predict(X_train)



# Report the before-and-afterscores

print("Unoptimized model\n------")

print("ROC-AUC on validation data: {:.4f}".format(roc_auc_score(y_val, predictions)))

print("\nOptimized Model\n------")

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_val, best_predictions_val)))

print("Final ROC-AUC on the training data: {:.4f}".format(roc_auc_score(y_train, best_predictions_train)))

print("Optimal parameters:", grid_obj.best_params_)
clf = AdaBoostClassifier(random_state = random_state)



# Create the parameters list you wish to tune, using a dictionary if needed.

parameters = { 'n_estimators': range(1000,1501,100)}



# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = GridSearchCV(clf, param_grid = parameters, scoring= scorer,  verbose=1, n_jobs =-1)



#  Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train, y_train)



# Get the estimator

best_clf = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train, y_train)).predict(X_val)

best_predictions_val = best_clf.predict(X_val)

best_predictions_train = best_clf.predict(X_train)



# Report the before-and-afterscores

print("Unoptimized model\n------")

print("ROC-AUC on validation data: {:.4f}".format(roc_auc_score(y_val, predictions)))

print("\nOptimized Model\n------")

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_val, best_predictions_val)))

print("Final ROC-AUC on the training data: {:.4f}".format(roc_auc_score(y_train, best_predictions_train)))

print("Optimal parameters:", grid_obj.best_params_)
clf = AdaBoostClassifier(random_state = random_state)



# Create the parameters list you wish to tune, using a dictionary if needed.

parameters = { 'n_estimators': range(2000,3001,200),

              'learning_rate': [0.5]}



# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = GridSearchCV(clf, param_grid = parameters, scoring= scorer,  verbose=1, n_jobs =-1)



#  Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train, y_train)



# Get the estimator

best_clf = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train, y_train)).predict(X_val)

best_predictions_val = best_clf.predict(X_val)

best_predictions_train = best_clf.predict(X_train)



# Report the before-and-afterscores

print("Unoptimized model\n------")

print("ROC-AUC on validation data: {:.4f}".format(roc_auc_score(y_val, predictions)))

print("\nOptimized Model\n------")

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_val, best_predictions_val)))

print("Final ROC-AUC on the training data: {:.4f}".format(roc_auc_score(y_train, best_predictions_train)))

print("Optimal parameters:", grid_obj.best_params_)
clf = AdaBoostClassifier(n_estimators=4000,random_state = random_state, learning_rate = 0.5)

clf.fit(X_train, y_train)

best_predictions_val_ab = clf.predict(X_val)

best_predictions_train_ab = clf.predict(X_train)

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_val, best_predictions_val_ab)))

print("Final ROC-AUC on the training data: {:.4f}".format(roc_auc_score(y_train, best_predictions_train_ab)))
#fit model with optimal parameters found during gridsearch:

clf_AB = AdaBoostClassifier(n_estimators=1200,random_state = random_state)

clf_AB.fit(X_train, y_train)

# predict outcome using predict_probe instead of predict function:

probs_train_ab = clf_AB.predict_proba(X_train)[:, 1]

probs_val_ab = clf_AB.predict_proba(X_val)[:, 1]

print("score train: {}".format(roc_auc_score(y_train, probs_train_ab)))

print("score validation: {}".format(roc_auc_score(y_val, probs_val_ab)))
# Initialize the classifier

clf = GradientBoostingClassifier(random_state = random_state)



# Create the parameters list you wish to tune, using a dictionary if needed.

parameters = {  'n_estimators': range(20,101,20),

                'learning_rate':[0.2],

                'min_samples_split': [500],

                'min_samples_leaf' : [50],

                'max_depth' : [8],

                'subsample' : [0.8]}



#Make an fbeta_score scoring object using make_scorer()

scorer = make_scorer(roc_auc_score)



# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = GridSearchCV(clf, param_grid = parameters, scoring= scorer,verbose=1, n_jobs =-1)



#  Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train, y_train)



# Get the estimator

best_clf = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train, y_train)).predict(X_val)

best_predictions_val = best_clf.predict(X_val)

best_predictions_train = best_clf.predict(X_train)



# Report the before-and-afterscores

print("Unoptimized model\n------")

print("ROC-AUC on validation data: {:.4f}".format(roc_auc_score(y_val, predictions)))

print("\nOptimized Model\n------")

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_val, best_predictions_val)))

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_train, best_predictions_train)))

print("Optimal parameters:", grid_obj.best_params_)
# Initialize the classifier

clf = GradientBoostingClassifier(random_state = random_state)



# Create the parameters list you wish to tune, using a dictionary if needed

parameters = {'max_depth':range(2,12,2), 

              'min_samples_split':range(100,601,100),

              'n_estimators': [80],

              'learning_rate':[0.2],                

              'min_samples_leaf' : [50],

              'subsample' : [0.8]

              }



#Make an fbeta_score scoring object using make_scorer()

scorer = make_scorer(roc_auc_score)



# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = GridSearchCV(clf, param_grid = parameters, scoring= scorer,  verbose=1, n_jobs =-1)



#  Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train, y_train)



# Get the estimator

best_clf = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train, y_train)).predict(X_val)

best_predictions_val = best_clf.predict(X_val)

best_predictions_train = best_clf.predict(X_train)



# Report the before-and-afterscores

print("Unoptimized model\n------")

print("ROC-AUC on validation data: {:.4f}".format(roc_auc_score(y_val, predictions)))

print("\nOptimized Model\n------")

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_val, best_predictions_val)))

print("Final ROC-AUC on the training data: {:.4f}".format(roc_auc_score(y_train, best_predictions_train)))

print("Optimal parameters:", grid_obj.best_params_)
# Initialize the classifier

clf = GradientBoostingClassifier(random_state = random_state)



# Create the parameters list you wish to tune, using a dictionary if needed.

parameters = {'min_samples_leaf':range(10,71,10),

              'max_depth': [6], 

              'min_samples_split': [200],

              'n_estimators': [80],

              'learning_rate':[0.2],

              'subsample' : [0.8]}



#Make an fbeta_score scoring object using make_scorer()

scorer = make_scorer(roc_auc_score)



# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = GridSearchCV(clf, param_grid = parameters, scoring= scorer, verbose=1, n_jobs =-1)





# Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train, y_train)



# Get the estimator

best_clf = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train, y_train)).predict(X_val)

best_predictions_val = best_clf.predict(X_val)

best_predictions_train = best_clf.predict(X_train)



# Report the before-and-afterscores

print("Unoptimized model\n------")

print("ROC-AUC on validation data: {:.4f}".format(roc_auc_score(y_val, predictions)))

print("\nOptimized Model\n------")

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_val, best_predictions_val)))

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_train, best_predictions_train)))

print("Optimal parameters:", grid_obj.best_params_)
# Initialize the classifier

clf = GradientBoostingClassifier(random_state = random_state)



# Create the parameters list you wish to tune, using a dictionary if needed.

parameters = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],

              'min_samples_leaf': [50],

              'max_depth': [6], 

              'min_samples_split': [200],

              'n_estimators': [80],

              'learning_rate':[0.2]}



#Make an fbeta_score scoring object using make_scorer()

scorer = make_scorer(roc_auc_score)



# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = GridSearchCV(clf, param_grid = parameters, scoring= scorer, verbose=1, n_jobs =-1)



# Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train, y_train)



# Get the estimator

best_clf = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train, y_train)).predict(X_val)

best_predictions_val = best_clf.predict(X_val)

best_predictions_train = best_clf.predict(X_train)



# Report the before-and-afterscores

print("Unoptimized model\n------")

print("ROC-AUC on validation data: {:.4f}".format(roc_auc_score(y_val, predictions)))

print("\nOptimized Model\n------")

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_val, best_predictions_val)))

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_train, best_predictions_train)))

print("Optimal parameters:", grid_obj.best_params_)
# Initialize the classifier

clf = GradientBoostingClassifier(random_state = random_state)



# Create the parameters list you wish to tune, using a dictionary if needed.

parameters = {'subsample':[0.8],

              'min_samples_leaf': [50],

              'max_depth': [6], 

              'min_samples_split': [200],

              'n_estimators': range(140, 241, 20),

              'learning_rate':[0.1]}



#Make an fbeta_score scoring object using make_scorer()

scorer = make_scorer(roc_auc_score)



# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = GridSearchCV(clf, param_grid = parameters, scoring= scorer, verbose=1, n_jobs =-1)



#  Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train, y_train)



# Get the estimator

best_clf = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train, y_train)).predict(X_val)

best_predictions_val = best_clf.predict(X_val)

best_predictions_train = best_clf.predict(X_train)



# Report the before-and-afterscores

print("Unoptimized model\n------")

print("ROC-AUC on validation data: {:.4f}".format(roc_auc_score(y_val, predictions)))

print("\nOptimized Model\n------")

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_val, best_predictions_val)))

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_train, best_predictions_train)))

print("Optimal parameters:", grid_obj.best_params_)
# Initialize the classifier

clf = GradientBoostingClassifier(random_state = random_state)



# Create the parameters list you wish to tune, using a dictionary if needed.

parameters = {'subsample':[0.8],

              'min_samples_leaf': [50],

              'max_depth': [6], 

              'min_samples_split': [200],

              'n_estimators': range(360, 401, 20) ,

              'learning_rate':[0.05]}



#Make an fbeta_score scoring object using make_scorer()

scorer = make_scorer(roc_auc_score)



# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = GridSearchCV(clf, param_grid = parameters, scoring= scorer, verbose=1, n_jobs =-1)



#  Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train, y_train)



# Get the estimator

best_clf = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train, y_train)).predict(X_val)

best_predictions_val = best_clf.predict(X_val)

best_predictions_train = best_clf.predict(X_train)



# Report the before-and-afterscores

print("Unoptimized model\n------")

print("ROC-AUC on validation data: {:.4f}".format(roc_auc_score(y_val, predictions)))

print("\nOptimized Model\n------")

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_val, best_predictions_val)))

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_train, best_predictions_train)))

print("Optimal parameters:", grid_obj.best_params_)
# fitting GradientBoostingClassifier with optimal parameters:

clf_GB = GradientBoostingClassifier(random_state = random_state, subsample = 0.8, min_samples_leaf = 50,

              max_depth = 6, min_samples_split = 200, n_estimators = 180, learning_rate = 0.1 )

clf_GB.fit(X_train, y_train)



best_predictions_val_gb = clf_GB.predict(X_val)

best_predictions_train_gb = clf_GB.predict(X_train)

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_val, best_predictions_val_gb)))

print("Final ROC-AUC on the training data: {:.4f}".format(roc_auc_score(y_train, best_predictions_train_gb)))
# predict outcome using predict_probe instead of predict function:

probs_train_gb = clf_GB.predict_proba(X_train)[:, 1]

probs_val_gb = clf_GB.predict_proba(X_val)[:, 1]

print("score train: {}".format(roc_auc_score(y_train, probs_train_gb)))

print("score test: {}".format(roc_auc_score(y_val, probs_val_gb)))
clf = LogisticRegression(random_state= random_state)



# Create the parameters list you wish to tune, using a dictionary if needed.

parameters = { 'C': [0.001, 0.01, 0.05, 0.1, 0.5, 0.7, 0.8, 0.9, 1, 5, 10, 20, 50]}



#Make an fbeta_score scoring object using make_scorer()

scorer = make_scorer(roc_auc_score)



# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = GridSearchCV(clf, param_grid = parameters, scoring= scorer,  verbose=1, n_jobs =-1)



#  Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train, y_train)



# Get the estimator

clf_LR = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train, y_train)).predict(X_val)

best_predictions_val_lr = clf_LR.predict(X_val)

best_predictions_train_lr = clf_LR.predict(X_train)



# Report the before-and-afterscores

print("Unoptimized model\n------")

print("ROC-AUC on validation data: {:.4f}".format(roc_auc_score(y_val, predictions)))

print("\nOptimized Model\n------")

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_val, best_predictions_val_lr)))

print("Final ROC-AUC on the training data: {:.4f}".format(roc_auc_score(y_train, best_predictions_train_lr)))

print("Optimal parameters:", grid_obj.best_params_)
# predict outcome using predict_probe instead of predict function:

probs_train_lr = clf_LR.predict_proba(X_train)[:, 1]

probs_val_lr = clf_LR.predict_proba(X_val)[:, 1]

print("score train: {}".format(roc_auc_score(y_train, probs_train_lr)))

print("score test: {}".format(roc_auc_score(y_val, probs_val_lr)))
clf = RandomForestClassifier(random_state= random_state)



# Create the parameters list you wish to tune, using a dictionary if needed.

parameters = { 'n_estimators': range(20,1020,100),

                'max_depth': range(2, 10, 1)}



#Make an fbeta_score scoring object using make_scorer()

scorer = make_scorer(roc_auc_score)



# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = GridSearchCV(clf, param_grid = parameters, scoring= scorer,  verbose=1, n_jobs =-1)



#  Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train, y_train)



# Get the estimator

clf_RF = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train, y_train)).predict(X_val)

best_predictions_val_rf = clf_RF.predict(X_val)

best_predictions_train_rf = clf_RF.predict(X_train)



# Report the before-and-afterscores

print("Unoptimized model\n------")

print("ROC-AUC on validation data: {:.4f}".format(roc_auc_score(y_val, predictions)))

print("\nOptimized Model\n------")

print("Final ROC-AUC on the validation data: {:.4f}".format(roc_auc_score(y_val, best_predictions_val_rf)))

print("Final ROC-AUC on the training data: {:.4f}".format(roc_auc_score(y_train, best_predictions_train_rf)))

print("Optimal parameters:", grid_obj.best_params_)
# predict outcome using predict_probe instead of predict function:

probs_train_rf = clf_RF.predict_proba(X_train)[:, 1]

probs_val_rf = clf_RF.predict_proba(X_val)[:, 1]

print("score train: {}".format(roc_auc_score(y_train, probs_train_rf)))

print("score test: {}".format(roc_auc_score(y_val, probs_val_rf)))
print("score train for Adaboost: {}".format(roc_auc_score(y_train, probs_train_ab)))

print("score test for Adaboost: {}".format(roc_auc_score(y_val, probs_val_ab)))

print("score train for Gradient Boosting: {}".format(roc_auc_score(y_train, probs_train_gb)))

print("score test for Gradient Boosting: {}".format(roc_auc_score(y_val, probs_val_gb)))

print("score train for Logistic Regression: {}".format(roc_auc_score(y_train, probs_train_lr)))

print("score test for Logistic Regression: {}".format(roc_auc_score(y_val, probs_val_lr)))

print("score train for Random Forest: {}".format(roc_auc_score(y_train, probs_train_rf)))

print("score test for Random Forest: {}".format(roc_auc_score(y_val, probs_val_rf)))

print("score train for top2 models: {}".format(roc_auc_score(y_train, (probs_train_gb+probs_train_ab)/2)))

print("score test for top2 models: {}".format(roc_auc_score(y_val, (probs_val_gb+probs_val_ab)/2)))

print("score train for top3 models: {}".format(roc_auc_score(y_train, (probs_train_gb+probs_train_ab+probs_train_rf)/3)))

print("score test for top3 models: {}".format(roc_auc_score(y_val, (probs_val_gb+probs_val_ab+probs_val_rf)/3)))

print("score train for all models: {}".format(roc_auc_score(y_train, (probs_train_gb+probs_train_ab+

                                                                      probs_train_rf+probs_train_lr)/4)))

print("score test for all models: {}".format(roc_auc_score(y_val, (probs_val_gb+probs_val_ab+

                                                                      probs_val_rf+ probs_val_lr)/4)))
# make a copy of data_test to overwrite duting feature engineering:

X_test = data_test[:]
X_test.info()
# fill missing values for numeric variables with approximatelly gaussian dictribution:

for col in ['age', 'education-num', 'hours-per-week']:

    X_test[col]= X_test[col].fillna(data_train[col].mean())



# fill missing values for numeric variables with skewed dictribution:

for col in ['capital-gain', 'capital-loss']:

    X_test[col]= X_test[col].fillna(data_train[col].median())



#fill missing categorical values with most freaquent category:

for col in categorical:

    X_test[col]= X_test[col].fillna(data_train.groupby([col])[col].count().sort_values(ascending=False).index[0])

    
#check for missing values in X_test after filling them in:

X_test.info()
#log transform skewed data

X_test[skewed] = log_transform(X_test)



#scale continues variables:

X_test[continues]= normalize(X_test)



# One-hot encode thedata using pandas.get_dummies()

X_test_final = pd.get_dummies(X_test)



# Print the number of features after one-hot encoding

encoded = list(X_test_final.columns)

print("{} total features after one-hot encoding.".format(len(encoded)))
best_model = clf_GB

test = pd.read_csv("../input/test_census.csv")



test['id'] = test.iloc[:,0] 

test['income'] = best_model.predict_proba(X_test_final)[:, 1]



test[['id', 'income']].to_csv("submissionGB.csv", index=False)
best_model = clf_AB

test = pd.read_csv("../input/test_census.csv")



test['id'] = test.iloc[:,0] 

test['income'] = best_model.predict_proba(X_test_final)[:, 1]



test[['id', 'income']].to_csv("submissionAB.csv", index=False)
best_model = clf_LR

test = pd.read_csv("../input/test_census.csv")



test['id'] = test.iloc[:,0] 

test['income'] = best_model.predict_proba(X_test_final)[:, 1]



test[['id', 'income']].to_csv("submissionLR.csv", index=False)
best_model = clf_RF

test = pd.read_csv("../input/test_census.csv")



test['id'] = test.iloc[:,0] 

test['income'] = best_model.predict_proba(X_test_final)[:, 1]



test[['id', 'income']].to_csv("submissionRF.csv", index=False)
test = pd.read_csv("../input/test_census.csv")



test['id'] = test.iloc[:,0] 

test['income'] = (clf_GB.predict_proba(X_test_final)[:, 1] + clf_AB.predict_proba(X_test_final)[:, 1])/2



test[['id', 'income']].to_csv("submission_top2.csv", index=False)
test = pd.read_csv("../input/test_census.csv")



test['id'] = test.iloc[:,0] 

test['income'] = (clf_GB.predict_proba(X_test_final)[:, 1] + clf_AB.predict_proba(X_test_final)[:, 1]+

                  clf_RF.predict_proba(X_test_final)[:, 1])/3



test[['id', 'income']].to_csv("submission_top3.csv", index=False)
test = pd.read_csv("../input/test_census.csv")



test['id'] = test.iloc[:,0] 

test['income'] = (clf_GB.predict_proba(X_test_final)[:, 1] + clf_AB.predict_proba(X_test_final)[:, 1]+

                clf_RF.predict_proba(X_test_final)[:, 1] + clf_LR.predict_proba(X_test_final)[:, 1])/4



test[['id', 'income']].to_csv("submission_all.csv", index=False)