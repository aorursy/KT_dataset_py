# Importing libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import os

import warnings



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, QuantileTransformer

from sklearn.utils import shuffle

from sklearn.ensemble import IsolationForest, RandomForestClassifier

from sklearn.svm import OneClassSVM, SVC

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix



from collections import Counter



pd.set_option('display.max_columns', None)

warnings.filterwarnings('ignore')
# Loading dataset

df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

print(df.shape)

print(df['Class'].value_counts())

df.head()
def make_data(dataset, scale = None, ballance = None):

    

    '''

    Creates train and validation datasets using differend scaling and undersampling methods

    

    scale - scaling method:

     - 'standard' for StandardScaler 

     - 'quantile' for QuantileTransformer

     - None - do not use scaling

     

    ballance - method of dataset undersampling:

     - 'random_smaples' - creates subset of random non fraud_trasactions with number of samples equal to fraud transactions amount

     - 'outlier elimination' - removing outliers from majority class with IF, then extracting support vectors using OCSVM 

        and then sub-sampling this subset using random samples

     - None - do not use ballancing

    '''

    

    data = dataset.copy()

        

    # Split data to train and validation datasets

    Y = data['Class']

    X = data.drop('Class', axis = 1)

    x, val_x, y, val_y = train_test_split(X, Y, test_size = 0.15, stratify = Y, random_state = 1)

        

    # Ballancing data if method is 'random_samples'

    if ballance == 'random_samples':        

        x['Class'] = y

        fraud = x[x['Class'] == 1]        

        non_fraud = x[x['Class'] == 0]        

        non_fraud = non_fraud.sample(fraud.shape[0], random_state = 1)        

        x = pd.concat([fraud, non_fraud])

        y = x['Class']

        x = x.drop('Class', axis = 1)

    

    # Scaling data

    if scale:        

        if scale == 'standard':            

            scaler = StandardScaler()        

            x = pd.DataFrame(scaler.fit_transform(x), index = x.index, columns = x.columns)

            val_x = pd.DataFrame(scaler.transform(val_x), index = val_x.index, columns = val_x.columns)            

        elif scale == 'quantile':            

            scaler = QuantileTransformer(output_distribution = 'normal')

            x = pd.DataFrame(scaler.fit_transform(x), index = x.index, columns = x.columns)

            val_x = pd.DataFrame(scaler.transform(val_x), index = val_x.index, columns = val_x.columns)            

        else:

            raise Exception(f'{scale} is not supported scaling method.\nUse "standard" or "quantile" methods.')        

    

    # Ballancing data if method is 'outlier_elimination'

    if ballance == 'outlier_elimination':        

        x['Class'] = y

        fraud = x[x['Class'] == 1]        

        non_fraud = x[x['Class'] == 0]

        non_fraud.drop('Class', axis = 1, inplace = True)        

        

        # Detecting outliers with IF

        IF = IsolationForest(n_jobs = -1)

        outliers = IF.fit_predict(non_fraud)

                

        # Removing outliers

        non_fraud['Outliers'] = outliers

        non_fraud = non_fraud[non_fraud['Outliers'] == 1]

        non_fraud.drop('Outliers', axis = 1, inplace = True)        

        

        # Taking no more than 25000 samples for OCSVM train

        if non_fraud.shape[0] > 25000:

            non_fraud = non_fraud.sample(25000, random_state = 1)

        

        # Training OCSVM        

        ocsvm = OneClassSVM(gamma = 'auto', kernel = 'rbf', max_iter = 10)

        ocsvm.fit(non_fraud)        

        # Extracting support vectors

        vectors = pd.DataFrame(ocsvm.support_vectors_, columns = non_fraud.columns)

        

        # Ballancing dataset taking random samples from support vectors        

        non_fraud = vectors.sample(fraud.shape[0], random_state = 1)        

        non_fraud['Class'] = 0

        x = pd.concat([fraud, non_fraud])

        y = x['Class']

        x = x.drop('Class', axis = 1)

        

    elif ballance not in ['random_samples', 'outlier_elimination'] and ballance is not None:

        raise Exception(f'{ballance} is not supported balancing method.\nUse "random_samples" or "outlier_elimination" methods.')

       

    x, y = shuffle(x, y)    

       

    return (x, val_x, y, val_y)
# Creating datasets for experiments

datasets = {

    'Raw_standard': make_data(df, scale = 'standard'),

    'Raw_quantile': make_data(df, scale = 'quantile'),

    'Random_samples_standard': make_data(df, scale = 'standard', ballance = 'random_samples'),

    'Random_samples_quantile': make_data(df, scale = 'quantile', ballance = 'random_samples'),

    'Out_elim_standard': make_data(df, scale = 'standard', ballance = 'outlier_elimination'),

    'Out_elim_quantile': make_data(df, scale = 'quantile', ballance = 'outlier_elimination'),

           }



# Defining classifiers

classifiers = {

    'SVM': SVC(random_state = 1),

    'RandomForest': RandomForestClassifier(n_jobs = -1, random_state = 1),

    'LogisticRegression': LogisticRegression(n_jobs = -1, random_state = 1),

              }
# Training classifiers and plotting results

for classifier in classifiers:        

    fig = plt.figure(figsize = (13, 10))

    

    for i, key in enumerate(datasets.keys()):

        print(f'Training {classifier}: {key}')

        

        # Train classifier

        clf = classifiers[classifier]

        clf.fit(datasets[key][0], datasets[key][2])

        preds = clf.predict(datasets[key][1])

        

        # Create confusin matrix plot

        plt.subplot(f'23{i+1}')

        cm = confusion_matrix(datasets[key][-1], preds)

        sns.heatmap(cm, fmt = 'd', annot = True, square = True, cbar = False, cmap = 'Blues', 

                    xticklabels =  ['P_Non_fraud', 'P_Fraud'], yticklabels = ['Non_fraud', 'Fraud']).set_title(key)

        

    print('\n', classifier)

    

    # Show plots

    plt.show()