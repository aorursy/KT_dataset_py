import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import seaborn as sns
def show_metrics(cm):

    tp = cm[1,1]

    fn = cm[1,0]

    fp = cm[0,1]

    tn = cm[0,0]

    print('Precision =     {:.3f}'.format(tp/(tp+fp)))

    print('Recall    =     {:.3f}'.format(tp/(tp+fn)))

    print('F1_score  =     {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/

                                                 ((tp/(tp+fp))+(tp/(tp+fn))))))
dataset = pd.read_csv('../input/creditcard.csv')

n_normal_transactions=len(dataset[dataset.Class==0])

n_fraud_transactions=len(dataset[dataset.Class==1])

print(n_normal_transactions)

print(n_fraud_transactions)

dataset.head()
data_train, data_test= train_test_split(dataset, test_size = 0.2, random_state = 0)
def over_sample(data_train):



    

    normal_transactions=data_train[data_train.Class==0]

    fraud_transactions=data_train[data_train.Class==1]

    

    resampled_data=normal_transactions

    

    for i in range(350):

        resampled_data=pd.concat([resampled_data,fraud_transactions])

        

    resampled_data=resampled_data.sample(frac=1)

    

    return resampled_data





#resampling data

data_train=over_sample(data_train)





n_fraud_transactions=len(data_train[data_train.Class==1])

n_normal_transactions=len(data_train[data_train.Class==0])



print(n_normal_transactions)

print(n_fraud_transactions)
data_train=data_train.drop('Time',1)

sc=StandardScaler()

data_train["Normalized Amount"] = sc.fit_transform(data_train['Amount'].values.reshape(-1, 1))

data_train.drop('Amount',axis=1,inplace=True)



y_train = data_train.iloc[:, 28].values

X_train = data_train.drop('Class',1).values





data_test=data_test.drop('Time',1)

data_test["Normalized Amount"] = sc.transform(data_test['Amount'].values.reshape(-1, 1))

data_test.drop('Amount',axis=1,inplace=True)



y_test = data_test.iloc[:, 28].values

X_test = data_test.drop('Class',1).values

from sklearn.ensemble import RandomForestClassifier



rf_cfl=RandomForestClassifier(n_estimators = 100,max_features= 2, min_samples_leaf= 1, min_samples_split= 5)

rf_cfl.fit(X_train, y_train)

y_pred=rf_cfl.predict(X_test)

from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_test, y_pred)



plt.figure(figsize = (12,10))

sns.heatmap(cm_rf, annot=True, cmap="coolwarm")



show_metrics(cm_rf)



#grid search



from sklearn.model_selection import GridSearchCV



param_grid = {

            'n_estimators': [100,200,300],

            'max_features': [2, 3],

            'min_samples_leaf': [1, 2, 4],

            'min_samples_split': [2, 5, 10]

            }



CV_rnd_cfl = GridSearchCV(estimator = rf_cfl, param_grid = param_grid, scoring = 'precision', verbose = 10, n_jobs = 1)

CV_rnd_cfl.fit(X_train, y_train)



best_parameters = CV_rnd_cfl.best_params_

print("The best parameters for using this model are", best_parameters)
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

log_cfl = LogisticRegression(C= 0.1, class_weight= 'balanced', penalty= 'l1')

log_cfl.fit(X_train, y_train)

y_pred = log_cfl.predict(X_test)

cm_log = confusion_matrix(y_test, y_pred)



plt.figure(figsize = (12,10))

sns.heatmap(cm_log, annot=True, cmap="coolwarm")



show_mertics(cm_log)





#grid search

from sklearn.model_selection import GridSearchCV

param_grid = {

            'penalty' : ['l1','l2'], 

            'class_weight' : ['balanced', None], 

            'C' : [0.1, 1, 10, 100]

            }

CV_log_cfl = GridSearchCV(estimator = log_cfl, param_grid = param_grid , scoring = 'recall', verbose = 1, n_jobs = 8)

CV_log_cfl.fit(X_train, y_train)



best_parameters = CV_log_cfl.best_params_

print('The best parameters for using this model are', best_parameters)

from sklearn.ensemble import VotingClassifier



voting_cfl = VotingClassifier (estimators = [('log', log_cfl), ('rf', rf_cfl),],voting='soft', weights = [1, 1.5])

voting_cfl.fit(X_train,y_train)

y_pred = voting_cfl.predict(X_test)



cm_tot=confusion_matrix(y_test,y_pred)



plt.figure(figsize = (12,10))

sns.heatmap(cm_tot, annot=True, cmap="coolwarm")



show_metrics(cm_tot)
