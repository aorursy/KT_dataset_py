import numpy as np 

import pandas as pd 

        

full_train= pd.read_csv('../input/human-activity-recognition-with-smartphones/train.csv', low_memory= False)

full_test= pd.read_csv('../input/human-activity-recognition-with-smartphones/test.csv', low_memory= False)

train_feats= full_train.iloc[:,:-2]  # last 2 columns contain subject number and label

train_labels= full_train.iloc[:,-1]

test_feats= full_test.iloc[:,:-2]

test_labels= full_test.iloc[:,-1]

print('Any NaNs in training set?', full_train.isnull().values.any())

print('Any NaNs in test set?', full_test.isnull().values.any())

print('Number of categorical features?', np.sum(train_feats.dtypes == 'category'))



print('Number of observations in training set:', np.shape(train_feats)[0])

print('Number of observations in test set:', np.shape(test_feats)[0])

print('Number of features:', np.shape(train_feats)[1])
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel



def important_feats(x_train, y_train, x_test):

    """

    Function that fits a random forest and extracts the importance weight from the forest for each feature to determine which features are most important

    (Features with an importance weight greater than 5x the median importance weight are most important)

    

    INPUTS: x_train is a pandas dataframe where each row is one example and each column is a feature (training data)

            y_train is a pandas dataframe with the corresponding labels to each example in x_train

            x_test is a pandas dataframe where each row is one example and each column is a feature (test data)

            

    OUTPUTS: x_train_new is the same as x_train except with only the most important features retained

            x_test_new is the same as x_test except with only the most important features retained

    """

    # define and fit tree

    forest= RandomForestClassifier(n_estimators= 500, random_state= 0)

    forest.fit(x_train, y_train)

    

    # select most important features

    selector= SelectFromModel(forest, threshold= 'median').fit(x_train, y_train)

    threshold= selector.threshold_

    selector= SelectFromModel(forest, threshold= 5*threshold).fit(x_train, y_train) # use 1.5x the median threshold

    important_feats= np.array([])

    

    for i in selector.get_support(indices= True):

        important_feats= np.append(important_feats, x_train.columns[i])

        

    # create new training and test sets that have only the most important features

    x_train_new= pd.DataFrame(selector.transform(x_train), columns= important_feats)

    x_test_new= pd.DataFrame(selector.transform(x_test), columns= important_feats)

    

    return important_feats, x_train_new, x_test_new



# run the above function to identify the names of the most important features and to perform feature selection (reduce the number of features to use in our ML models)

keep_feats, x_train_new, x_test_new= important_feats(train_feats, train_labels, test_feats)



feat_count= np.shape(keep_feats)[0]

print('Number of features after performing feature selection:', feat_count)
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')



# 21 subjects are in the training set. for leave-one-subject-out cross validation, a model is trained on all but one of the training subjects and the model performance on the remaining subject is recorded. this process repeats until all subjects have had an opportunity to be in the validation set 

train_subject_id= full_train.iloc[:,-2].unique() # get subject IDs

train_new= x_train_new.copy()

train_new['subject']= full_train.iloc[:,-2] # add back in subject IDs to training data

train_new['label']= full_train.iloc[:,-1] # add back in labels to training data



# create placeholder to store grid search results:

grid_logreg= np.zeros((6, np.shape(train_subject_id)[0])) # 6 hyperparameter setting combos are being tested



def loo_split(train_matrix, subject_id):

    """ Function that splits and standardizes training data into training and validation sets on a leave-one-subject-out basis

    INPUTS: train_matrix is a pd dataframe where all columns except the last two are feature values. the second last column is the subject ID and the last column is the label for the example

            subject_id is an integer identifying which subject is to make up the validation set

    OUTPUTS: train_scaled is a np array that contains all normalized feature values for the training set

            validation_scaled is a np array that contains all normalized feature values for the validation set

            y_train is a np array that contains the labels corresponding to the examples in train_scaled

            validation_labels is a np array that contains the labels corresponding to the examples in validation_scaled

    """

    validation_feats= train_matrix.iloc[:,:-2].loc[train_matrix['subject'] == subject_id]

    validation_labels= train_matrix.iloc[:,-1].loc[train_matrix['subject'] == subject_id]

    x_train= train_matrix.iloc[:,:-2].loc[train_matrix['subject'] != subject_id]

    y_train= train_matrix.iloc[:,-1].loc[train_matrix['subject'] != subject_id]

    

    # normalize feature values to have a mean= 0 and variance= 1

    scaler= StandardScaler().fit(x_train)

    train_scaled= scaler.transform(x_train)

    validation_scaled= scaler.transform(validation_feats)

    

    return train_scaled, validation_scaled, y_train, validation_labels





j= 0

for i in train_subject_id:

    k= 0

    train_scaled, validation_scaled, y_train, validation_labels= loo_split(train_new, i)

    

    for reg in [0.3, 0.6, 1.0]: # hyperparameter grid search for regularization strength

        for max_iter in [100, 500]: # hyperparameter grid search for max iterations

            lr_model= LogisticRegression(C= reg, max_iter= max_iter, random_state= 0).fit(train_scaled, y_train)

            grid_logreg[k,j]= lr_model.score(validation_scaled, validation_labels)

            k += 1

    j += 1



# rows in avg_recall are the hyperparameter settings, columns are average recall for each subject as the validation set

validation_results= pd.DataFrame(data= grid_logreg, index= ['C= 0.3, iter= 100', 'C= 0.3, iter= 500', 'C= 0.6, iter= 100', 'C= 0.6, iter= 500', 'C= 1.0, iter= 100','C= 1.0, iter= 500'], columns= train_subject_id.astype(str))

print('Avg accuracy for each hyperparameter setting:','\n', validation_results.mean(axis= 1))
from sklearn.metrics import recall_score, precision_score



def scale(x_train, x_test):

    """ Function that scales all feature values in the training set to have mean= 0 and variance= 1 and scales the test set feature values using the same transformations from the test set

    INPUTS: x_train and x_test are both pd dataframes containing the feature values for the training and test sets respectively

    OUTPUTS: x_train_scaled and x_test_scaled are np arrays containing the scaled feature values for the training and test sets respectively

    """

    scaler= StandardScaler().fit(x_train)

    x_train_scaled= scaler.transform(x_train)

    x_test_scaled= scaler.transform(x_test)

    return x_train_scaled, x_test_scaled



def model_results(truth, predicts):

    """ Function that displays the precision and recall values for the test set for each of the classes in a multiclass classification task 

    INPUTS: truth is a pd series containing the ground truth label for all examples in the test set

            predicts is a np array containing the predicted label for all examples in the test set (in the same order as the ground truth labels)

    OUTPUTS: None (displays the precision and recall values in a table)

    """

    classes= truth.unique()

    recall= recall_score(truth, predicts, average= None)

    precision= precision_score(truth, predicts, average= None)

    results= pd.DataFrame(data= np.array([recall, precision]), index= ['Recall', 'Precision'], columns= classes)

    print(results)

    return

    

x_train_scaled, x_test_scaled= scale(x_train_new, x_test_new)



# train model using all training data and C= 0.3

lr_model= LogisticRegression(C= 1.0, max_iter= 500, random_state= 0).fit(x_train_scaled, train_labels)



# get predictions for training set

test_predicts= lr_model.predict(x_test_scaled)



# get recall and precision for each of the classes

model_results(test_labels, test_predicts)
from sklearn.neural_network import MLPClassifier



# create placeholder to store grid search results:

grid_ann= np.zeros((6, np.shape(train_subject_id)[0])) # 6 hyperparameter setting combos are being tested



j= 0

for i in train_subject_id:

    k= 0

    train_scaled, validation_scaled, y_train, validation_labels= loo_split(train_new, i)

    

    for hidden in [(10,), (25,), (50,)]: # hyperparameter grid search for number of hidden units

        for alpha in [0.0001, 0.001]: # hyperparameter grid search for regularization strength

            ann_model= MLPClassifier(hidden_layer_sizes= hidden, alpha= alpha, random_state= 0).fit(train_scaled, y_train)

            grid_ann[k,j]= ann_model.score(validation_scaled, validation_labels)

            k += 1

    j += 1



# rows in avg_recall are the hyperparameter settings, columns are average recall for each subject as the validation set

validation_results= pd.DataFrame(data= grid_ann, index= ['units= 10, alpha= 0.0001', 'units= 10, alpha= 0.001', 'units= 25, alpha= 0.0001', 'units= 25, alpha= 0.001', 'units= 50, alpha= 0.0001','units= 50, alpha= 0.001'], columns= train_subject_id.astype(str))

print('Avg accuracy for each hyperparameter setting:','\n', validation_results.mean(axis= 1))
x_train_scaled, x_test_scaled= scale(x_train_new, x_test_new)



# train model using all training data and C= 0.3

ann_model= MLPClassifier(hidden_layer_sizes= 25, alpha= 0.0001, random_state= 0).fit(x_train_scaled, train_labels)



# get predictions for training set

test_predicts= ann_model.predict(x_test_scaled)



# get recall and precision for each of the classes

model_results(test_labels, test_predicts)