# Data Manipulattion

import numpy as np

import pandas as pd



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Importing Dependencies

%matplotlib inline



# Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')
# Read and preview the train data from csv file.

train = pd.read_csv('../input/train.csv')

train.head(3)
# Read and preview the test data from csv file.

test = pd.read_csv('../input/test.csv')

train.head(3)
# Merge train and test data together. This eliminates the hassle of handling train and test data seperately for various analysis.

merged = pd.concat([train,test], sort = False)

merged.head(3)
# Let's see the shape of the combined data

merged.shape
# variable in the combined data

merged.columns
# data types of different variables

merged.info()
# Description of the data variables

merged.describe()
# Visualization of Missing variables

plt.figure(figsize=(8,4))

sns.heatmap(merged.isnull(), yticklabels=False, cbar=False, cmap='plasma')
# Count of missing variables

merged.isnull().sum()
# let's preview the cabin again.

merged['Cabin'].head()
# we see that Cabin contains some missing values. let's count it again.

merged['Cabin'].isnull().sum()
# Let's manully understand the Cabin column.

merged['Cabin'].value_counts().head()
# let's fill all NaNs of cabin as 'X'

merged['Cabin'].fillna(value = 'X', inplace=True)
# Keeping 1st charater from the Cabin

merged['Cabin'] = merged['Cabin'].apply(lambda x: x[0])

merged['Cabin'].value_counts()
#Let's see the Name column.

merged['Name'].head(10)
# Extracting title from Name and create a new variable Title.

merged['Title'] = merged['Name'].str.extract('([A-Za-z]+)\.')

merged['Title'].head()
# let's see the different categories of Title from Name column.

merged['Title'].value_counts()
# Replacing  Dr, Rev, Col, Major, Capt with 'Officer'

merged['Title'].replace(to_replace = ['Dr', 'Rev', 'Col', 'Major', 'Capt'], value = 'Officer', inplace=True)



# Replacing Dona, Jonkheer, Countess, Sir, Lady with 'Aristocrate'

merged['Title'].replace(to_replace = ['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'], value = 'Aristocrat', inplace = True)



#  Replace Mlle and Ms with Miss. And Mme with Mrs.

merged['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace = True)
# let's see how Tittle looks now

merged['Title'].value_counts()
# Merging Sibsp and Parch and creating new variable called 'Family_size'

merged['Family_size'] = merged.SibSp + merged.Parch + 1  # Adding 1 for single person

merged['Family_size'].value_counts()
# Create buckets of single, small, medium, and large and then put respective values into them.

merged['Family_size'].replace(to_replace = [1], value = 'single', inplace = True)

merged['Family_size'].replace(to_replace = [2,3], value = 'small', inplace = True)

merged['Family_size'].replace(to_replace = [4,5], value = 'medium', inplace = True)

merged['Family_size'].replace(to_replace = [6, 7, 8, 11], value = 'large', inplace = True)
# let's see how 'Family_size' looks now

merged['Family_size'].value_counts()
# let's preview the Ticket variable.

merged['Ticket'].head(10)
# Assign N if there is only number and no character. If there is a character, extract the character only.

ticket = []

for x in list(merged['Ticket']):

    if x.isdigit():

        ticket.append('N')

    else:

         ticket.append(x.replace('.','').replace('/','').strip().split(' ')[0])

# Swap values

merged['Ticket'] = ticket
# Let's count the categories in  Ticket

merged['Ticket'].value_counts()
# Keeping only the 1st character to reduce the Ticket categories

merged['Ticket'] = merged['Ticket'].apply(lambda x : x[0])

merged['Ticket'].value_counts()
# Create a function to count total outliers.

def outliers(variable):

    global filtered # Global keyword is used inside a function only when we want to do assignments or when we want to change a variable.

    

    # Calculate 1st, 3rd quartiles and iqr.

    q1, q3 = variable.quantile(0.25), variable.quantile(0.75)

    iqr = q3 - q1

    

    # Calculate lower fence and upper fence for outliers

    l_fence, u_fence = q1 - 1.5*iqr , q3 + 1.5*iqr   # Any values less than l_fence and greater than u_fence are outliers.

    

    # Observations that are outliers

    outliers = variable[(variable<l_fence) | (variable>u_fence)]

    print('Total Outliers of', variable.name,':', outliers.count())

    

    # Drop obsevations that are outliers

    filtered = variable.drop(outliers.index, axis = 0)
# Total number of outliers in Fare

outliers(merged['Fare'])
# Visualisation of Fare distribution with outliers

plt.figure(figsize=(13, 2))

sns.boxplot(x=merged["Fare"],palette='Blues')

plt.title('Fare distribution with outliers', fontsize=15 )
# Visualisation of Fare distribution without outliers

plt.figure(figsize=(13, 2))

sns.boxplot(x=filtered,palette='Blues')

plt.title('Fare distribution without outliers', fontsize=15 )
# Total number of outliers in Age

outliers(merged['Age'])
# Visualisation of Age distribution with outliers

plt.figure(figsize=(13, 2))

sns.boxplot(x=merged["Age"],palette='Blues')

plt.title('Age distribution with outliers', fontsize=15)
# Visualisation of Age distribution without outliers

plt.figure(figsize=(13, 2))

sns.boxplot(x=filtered,palette='Blues')

plt.title('Age distribution without outliers', fontsize=15)
# let's count the missing values for each variable

merged.isnull().sum()
# imputing Embarked with mode because Embarked is a categorical variable.

merged['Embarked'].value_counts()
# Here S is the most frequent

merged['Embarked'].fillna(value = 'S', inplace = True)
# Impute missing values of Fare. Fare is a numerical variable with outliers. Hence it will be imputed by median.'''

merged['Fare'].fillna(value = merged['Fare'].median(), inplace = True)
# Let's plot correlation heatmap to see which variable is highly correlated with Age. We need to convert categorical variable into numerical to plot correlation heatmap. So convert categorical variables into numerical.

df = merged.loc[:, ['Sex', 'Pclass', 'Embarked', 'Title', 'Family_size', 'Parch', 'SibSp', 'Cabin', 'Ticket']]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df = df.apply(le.fit_transform) # data is converted.

df.head(2)
 # Inserting Age in variable correlation.

df['Age'] = merged['Age']

# Move Age at index 0.

df = df.set_index('Age').reset_index()

df.head(2)
# Now create the heatmap correlation of df

plt.figure(figsize=(10,6))

sns.heatmap(df.corr(), cmap ='BrBG',annot = True)

plt.title('Variables correlated with Age')

plt.show()
# Create a boxplot to view the correlated and medium of the Pclass and Title variables with Age.

# Boxplot b/w Pclass and Age

sns.boxplot(y='Age', x='Pclass', data=merged)
# Boxplot b/w Title and Age

sns.boxplot(y='Age', x='Title', data=merged)
# Impute Age with median of respective columns (i.e., Title and Pclass)

merged['Age'] = merged.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
# let's check the missing value again.

merged.isnull().sum()
# Creating bin categories for Age 

label_names = ['infant', 'child', 'teenager','young_adult', 'adult', 'aged']



# Create range for each bin categrories of age

cut_points = [0,5,12,18,35,60,81]



#Create and view categorized Age with original Age.

merged['Age_binned'] = pd.cut(merged['Age'], cut_points, labels = label_names)



#Age with Categorized Age.

merged[['Age', 'Age_binned']].head(2)
# Create bin categories for Fare

groups = ['low','medium','high','very_high']



# Create range for each bin categories of Fare

cut_points = [-1, 130, 260, 390, 520]



#Create and view categorized Fare with original Fare

merged['Fare_binned'] = pd.cut(merged.Fare, cut_points, labels = groups)



# Fare with Categorized Fare

merged[['Fare', 'Fare_binned']].head(2)
# import scaling model

#from sklearn.preprocessing import MinMaxScaler



#Create a scaler object

#scaler = MinMaxScaler()



# Fit and transform the merged['Fare']

#merged['Fare'] = scaler.fit_transform(merged['Fare'].values.reshape(-1,1))

#merged['Fare'].head()
# checking the data type

merged.dtypes
# Correcting data types, converting into categorical variables.

merged.loc[:, ['Pclass', 'Sex', 'Embarked', 'Cabin', 'Title', 'Family_size', 'Ticket']] = merged.loc[:, ['Pclass', 'Sex', 'Embarked', 'Cabin', 'Title', 'Family_size', 'Ticket']].astype('category')



# Due to merging there are NaN values in Survived for test set observations.

merged['Survived'] = merged['Survived'].dropna().astype('int') #Converting without dropping NaN throws an error
# Check if data types have been corrected

merged.dtypes
# let's see all the variables

merged.head(2)
# droping the feature that would not be useful anymore

merged.drop(columns = ['Name', 'Age','SibSp', 'Parch','Fare'], inplace = True, axis = 1)

merged.columns
# convert categotical data into dummies variables

merged = pd.get_dummies(merged, drop_first=True)

merged.head(2)
#Let's split the train and test set to feed machine learning algorithm.

train = merged.iloc[:891, :]

test  = merged.iloc[891:, :]
#Drop passengerid from train set and Survived from test set.'''

train = train.drop(columns = ['PassengerId'], axis = 1)

test = test.drop(columns = ['Survived'], axis = 1)
# setting the data as input and output for machine learning models

X_train = train.drop(columns = ['Survived'], axis = 1) 

y_train = train['Survived']



# Extract test set

X_test  = test.drop("PassengerId", axis = 1).copy()
# See the dimensions of input and output data set.'''

print('Input Matrix Dimension:  ', X_train.shape)

print('Output Vector Dimension: ', y_train.shape)

print('Test Data Dimension:     ', X_test.shape)
# Now initialize all the classifiers object.



#1.Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()



#2.KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()



#3.Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state = 40)



#4.Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state = 40, n_estimators = 100)



#5.Support Vector Machines

from sklearn.svm import SVC

svc = SVC(gamma = 'auto')



#6. XGBoost 

from xgboost import XGBClassifier

xgb = XGBClassifier(n_job = -1, random_state = 40)
# Create a function that returns train accuracy of different models.



def train_accuracy(model):

    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)

    train_accuracy = np.round(train_accuracy*100, 2)

    return train_accuracy

    

# making the summary table of train accuracy.

train_accuracy = pd.DataFrame({'Train_accuracy(%)':[train_accuracy(lr), train_accuracy(knn), train_accuracy(dt), train_accuracy(rf), train_accuracy(svc), train_accuracy(xgb)]})

train_accuracy.index = ['LR', 'KNN','DT', 'RF', 'SVC', 'XGB']

sorted_train_accuracy = train_accuracy.sort_values(by = 'Train_accuracy(%)', ascending = False)



#Training Accuracy of the Classifiers

sorted_train_accuracy
# Create a function that returns mean cross validation score for different models.

def val_score(model):

    from sklearn.model_selection import cross_val_score

    val_score = cross_val_score(model, X_train, y_train, cv = 10, scoring = 'accuracy').mean()

    val_score = np.round(val_score*100, 2)

    return val_score



# making the summary table of cross validation accuracy.

val_score = pd.DataFrame({'val_score(%)':[val_score(lr), val_score(knn), val_score(dt), val_score(rf), val_score(svc), val_score(xgb)]})

val_score.index = ['LR', 'KNN','DT', 'RF', 'SVC', 'XGB']

sorted_val_score = val_score.sort_values(by = 'val_score(%)', ascending = False)



#cross validation accuracy of the Classifiers

sorted_val_score
# define all the model hyperparameters one by one first



# 1. For logistic regression

lr_params = {'penalty':['l1', 'l2'],

             'C': np.logspace(0, 2, 4, 8 ,10)}



# 2. For KNN

knn_params = {'n_neighbors':[4,5,6,7,8,9,10],

              'weights':['uniform', 'distance'],

              'algorithm':['auto', 'ball_tree','kd_tree','brute'],

              'p':[1,2]}



# 3. For DT

dt_params = {'max_features': ['auto', 'sqrt', 'log2'],

             'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 

             'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],

             'random_state':[46]}

# 4. For RF

rf_params = {'criterion':['gini','entropy'],

             'n_estimators':[ 10, 30, 200, 400],

             'min_samples_leaf':[1, 2, 3],

             'min_samples_split':[3, 4, 6, 7], 

             'max_features':['sqrt', 'auto', 'log2'],

             'random_state':[46]}

# 5. For SVC

svc_params = {'C': [0.1, 1, 10,100], 

              'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],

              'gamma': [ 1, 0.1, 0.001, 0.0001]}



#6. For XGB

xgb_params = xgb_params_grid = {'min_child_weight': [1, 5],

                   'gamma': [0.04, 0, 0.1, 1.5],

                   'subsample': [0.6, 0.8, 1.0],

                   'colsample_bytree': [0.46, 1.0],

                   'max_depth': [3, 7]}
# Create a function to tune hyperparameters of the selected models.

def tune_hyperparameters(model, param_grid):

    from sklearn.model_selection import GridSearchCV

    global best_params, best_score #if you want to know best parametes and best score

    

    # Construct grid search object with 10 fold cross validation.

    grid = GridSearchCV(model, param_grid, verbose = 3, cv = 10, scoring = 'accuracy', n_jobs = -1)

    # Fit using grid search.

    grid.fit(X_train, y_train)

    best_params, best_score = grid.best_params_, np.round(grid.best_score_*100, 2)

    return best_params, best_score
# Appling tune hyperparameters in the created funtion



# Tune LR hyperparameters.

tune_hyperparameters(lr, param_grid=lr_params)

lr_best_params, lr_best_score =  best_params, best_score

print('LR Best Score:', lr_best_score)

print('And Best Parameters:', lr_best_params)
# Tune KNN hyperparameters

tune_hyperparameters(knn, param_grid=knn_params)

knn_best_params, knn_best_score =  best_params, best_score
# Tune DT hyperparameters

tune_hyperparameters(dt, param_grid=dt_params)

dt_best_params, dt_best_score =  best_params, best_score
# Tune RF hyperparameters

tune_hyperparameters(rf, param_grid=rf_params)

rf_best_params, rf_best_score =  best_params, best_score
# Tune SVC hyperparameters

tune_hyperparameters(svc, param_grid=svc_params)

svc_best_params, svc_best_score =  best_params, best_score
# Tune XGB hyperparameters

xgb_opt = XGBClassifier(learning_rate = 0.04, n_estimators = 500, 

                       silent = 1, nthread = -1, random_state = 101)

tune_hyperparameters(xgb_opt, param_grid=xgb_params)

xgb_best_params, xgb_best_score =  best_params, best_score
# lets compares cross validation scores with tunned scores for different models.

# Create a dataframe of tunned scores and sort them in descending order.'''

tunned_scores = pd.DataFrame({'Tunned_accuracy(%)': [lr_best_score, knn_best_score, dt_best_score, rf_best_score, svc_best_score, xgb_best_score]})

tunned_scores.index = ['LR', 'KNN', 'DT', 'RF', 'SVC', 'XGB']

sorted_tunned_scores = tunned_scores.sort_values(by = 'Tunned_accuracy(%)', ascending = False)

# Models Accuracy after Optimization

sorted_tunned_scores
# Instantiate the models with optimized hyperparameters.

lr  = LogisticRegression(**lr_best_params)

knn = KNeighborsClassifier(**knn_best_params)

dt  = DecisionTreeClassifier(**dt_best_params)

rf  = RandomForestClassifier(**rf_best_params)

svc = SVC(**svc_best_params)

xgb = XGBClassifier(**xgb_best_params)
# Train all the models with optimised hyperparameters

models = { 'LR': lr, 'KNN':knn,'DT':dt,'RF':rf, 'SVC':svc, 'XGB':xgb}



# 10-fold Cross Validation after Optimization

score = []

for x, (keys, items) in enumerate(models.items()):

    # Train the models with optimized parameters using cross validation.

    # No need to fit the data. cross_val_score does that for us.

    # But we need to fit train data for prediction in the follow session.

    from sklearn.model_selection import cross_val_score

    items.fit(X_train, y_train)

    scores = cross_val_score(items, X_train, y_train, cv = 10, scoring = 'accuracy')*100

    score.append(scores.mean())

    print('Mean Accuracy: %0.4f (+/- %0.4f) [%s]'  % (scores.mean(), scores.std(), keys))
# Make prediction using all the trained models

model_prediction = pd.DataFrame({'LR':lr.predict(X_test), 'KNN':knn.predict(X_test), 'DT':dt.predict(X_test),'RF':rf.predict(X_test), 'SVC':svc.predict(X_test), 'XGB': xgb.predict(X_test)})



#All the Models Prediction 

model_prediction.head()
# Submission with the most accurate random forest classifier

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": rf.predict(X_test)})

submission.to_csv('submission_rf.csv', index = False)





# Submission with the most accurate SVC classifier.

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": svc.predict(X_test)})

submission.to_csv('submission_svc.csv', index = False)



# Submission with the most accurate XGB classifier.

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": xgb.predict(X_test)})

submission.to_csv('submission_xgb.csv', index = False)
# Example : How hard voting works

data =[[1, 1, 1, 0, 1],

       [0, 0, 0, 1, 0]]

display(pd.DataFrame(data, columns = ['Class', 'RF', 'LR', 'KNN', 'Hard_voting']).set_index('Class'))
#Create a data frame to store base modles prediction 

base_prediction = model_prediction # we have already make the data frame above of all the models prediction



base_prediction.head()
#Let's visualize the correlations among the predictions of base models.

plt.figure(figsize = (15,8))

sns.heatmap(base_prediction.corr(), annot=True)

plt.title('Prediction correlation ammong the Base Models', fontsize = 18)
# We will use mlxtend library to train, predict and plot decision regions of hard voting ensemble classifier

# Define base models for hard voting ensemble.

base_models = [lr, knn, dt, rf, xgb]



# Import ensemble classifier from mlxtend

from mlxtend.classifier import EnsembleVoteClassifier



# Initialize hard voting ensemble

hard_evc = EnsembleVoteClassifier(clfs= base_models, voting = 'hard')

print('Training Hard Voting Emsemble Classification')

display(hard_evc.fit(X_train, y_train))

print('-----Done-----')



# Predict with hard voting ensemble.

y_pred_hard_ecv = pd.DataFrame(hard_evc.predict(X_test), columns = ['HARD_ECV'])



# Hard voting cross validation score.

print('\nComputing Hard Voting Cross Val Score')

hard_x_val_score = cross_val_score(hard_evc, X_train, y_train, cv=10,  scoring = 'accuracy')

hard_x_val_score = np.round(hard_x_val_score.mean()*100,2)

print('----Done----')



# Compare hard voting score with best base models scores.

hard_vs_base_score = pd.DataFrame({'Hard_vs_base_score(%)': [hard_x_val_score, lr_best_score, knn_best_score, dt_best_score, rf_best_score, xgb_best_score]})

hard_vs_base_score.index = ['HARD_VAL_SCORE', 'LR', 'KNN', 'DT', 'RF', 'XGB']

display(hard_vs_base_score)
# See base models prediction with hard voting prediction.

df_hard_base = pd.concat([base_prediction.drop('SVC', axis=1),y_pred_hard_ecv], sort = False, axis = 1)

display(df_hard_base.head(7))
# Example: how soft voting works

data = [[0.49, 0.99, 0.49, 0.66, 1],

        [0.51, 0.01, 0.51, 0.34, 0]]

display(pd.DataFrame(data, columns=['RF', 'LR', 'KNN', 'Average', 'Soft Voting']))
# Base models for soft voting is the base models of hard voting

# Initialize soft voting ensemble

base_model = [lr, knn, dt, rf, xgb]

soft_evc = EnsembleVoteClassifier(clfs = base_model, voting = 'soft')

print('fitting soft voting ensemble')

display(soft_evc.fit(X_train, y_train))



#Predict with soft voting ensemble

y_pred_soft_evc = pd.DataFrame(soft_evc.predict(X_test), columns = ['SOFT_EVC'])



# Hard voting cross validation score

print('\nComputing Soft Voting X Val Score...')

soft_x_val_score = cross_val_score(soft_evc, X_train, y_train, cv = 10, scoring = 'accuracy')

soft_x_val_score = np.round(soft_x_val_score.mean()*100, 2)

print('----Done----')



# Compare Soft voting score with best base models scores.

soft_vs_base_score = pd.DataFrame({'Soft_Vs_Base_Score': [soft_x_val_score, lr_best_score, knn_best_score, dt_best_score, rf_best_score, xgb_best_score]})

soft_vs_base_score.index = ['SOFT_VAL_SCORE', 'LR', 'KNN', 'DT', 'RF', 'XGB']

display(hard_vs_base_score)
# Initialize bagging classifier

from sklearn.ensemble import BaggingClassifier

bagg = BaggingClassifier(base_estimator = rf, verbose = 0, n_jobs = -1, random_state = 45)

print('Fitting Bagging Ensemble')

display(bagg.fit(X_train, y_train))

print('---Done----')



# Bagging cross validation score.

print('\nComputing Bagging X Val Score..')

bagg_x_val_score = cross_val_score(bagg, X_train, y_train, cv = 10, scoring = 'accuracy')

bagg_x_val_score = np.round(bagg_x_val_score.mean()*100, 2)

print('----Done----')



# Compare bagging ensemble score with best base models scores

bagg_vs_base_score = pd.DataFrame({'Bagging_vs_Base_Score': [bagg_x_val_score,lr_best_score, knn_best_score, dt_best_score, rf_best_score, xgb_best_score]})

bagg_vs_base_score.index = ['BAGG', 'LR', 'KNN', 'DT', 'RF', 'XGB']

display(bagg_vs_base_score)
# Perform blending in mlens

from mlens.ensemble import BlendEnsemble



# Initialize blend ensembler

blend = BlendEnsemble(n_jobs = -1, test_size = 0.5, random_state = 45)



# Base models for blending.

base_models = [rf, dt, knn, xgb]

blend.add(base_models)



# Meta learner for blending. We will use lr.'''

blend.add_meta(lr)



# Train the blend ensemble.

print('Fitting Blending...')

display(blend.fit(X_train, y_train))

print('----Done----')
# Import stacking method from vecstack

from vecstack import stacking

from sklearn.metrics import accuracy_score



# Initialize base models. We will use the same base models as blending

base_models = [rf, dt, xgb, knn]



# Perform stacking

S_train, S_test = stacking(base_models,                # list of base models

                           X_train, y_train, X_test,   # data

                           regression = False,         # classification task (if you need 

                                                       # regression - set to True)

                           mode = 'oof_pred_bag',      # mode: oof for train set, predict test 

                                                       # set in each fold and vote

                           needs_proba = False,        # predict class labels (if you need 

                                                       # probabilities - set to True) 

                           save_dir = None,            # do not save result and log (to save 

                                                       # in current dir - set to '.')

                           metric = accuracy_score,    # metric: callable

                           n_folds = 10,               # number of folds

                           stratified = True,          # stratified split for folds

                           shuffle = True,             # shuffle the data

                           random_state = 45,          # ensure reproducibility

                           verbose = 1)                # print progress
print('Input features for meta learner')

display(S_train[:7])



print('Test/output (prediction set for meta learner')

display(S_test[:7])
# Dimension of S_train and S_test

print('Dimension of S_train:', S_train.shape)

print('Dimension of S_test:', S_test.shape)

# Initialize 1st level model that is our meta learner. We will use lr

super_learner = lr 

    

# Fit meta learner on the output of base learners

print('Fitting Stacking...')

super_learner.fit(S_train, y_train)



print('Done.')

# Finally predict using super learner.

y_pred_super = super_learner.predict(S_test)
# Predicting with different ensembles



# Hard voting

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": hard_evc.predict(X_test)})

submission.to_csv('submission_hard_evc.csv', index = False)



# Soft voting

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": soft_evc.predict(X_test)})

submission.to_csv('submission_soft_evc.csv', index = False)



# Bagging

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": bagg.predict(X_test)})

submission.to_csv('submission_bagg.csv', index = False)



# Blending

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": blend.predict(X_test).astype(int)})

submission.to_csv('submission_blend.csv', index = False)



# Stacking

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred_super.astype(int)})

submission.to_csv('submission_super.csv', index = False)