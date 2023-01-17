# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Data Analysis

import numpy as np

import pandas as pd



#Visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline





#For missing values

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



#Warnings

import warnings

warnings.filterwarnings('ignore')



#Preprocessing

from sklearn import preprocessing



#Scaling

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler





#Machine learning 

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import statsmodels.api as sm



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold



#Random forest

from sklearn.ensemble import RandomForestClassifier





#Vif

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Lets import the train and test data and look into it



train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_valid = pd.read_csv("/kaggle/input/titanic/test.csv")



merged = [train_df, test_df]



train_df.head()
# Looking into the datatype and count 



train_df.info()

print('-'*50)

test_df.info()
#Statistical Summary for numeric columns



train_df.describe()
#Statistical Summary for object columns columns



train_df.describe(include=['O'])
# Shape of dataset



print("Size of training data:{0}".format(train_df.shape))



print("Size of test data:{0}".format(test_df.shape))
# Let's look into the missing value percentage 



print(round(100*(train_df.isnull().sum()/len(train_df)),2))

print('-'*40)

print(round(100*(test_df.isnull().sum()/len(test_df)),2))
plt.figure(figsize = (10,8))



sns.heatmap(train_df.corr(), annot= True, cmap = 'YlGnBu')
# we will drop Cabin Column from the data has it contain lot of missing values





train_df.drop('Cabin', inplace = True, axis = 1)

test_df.drop('Cabin', inplace = True, axis =1)
# Since we assume that the sex and Pclass is important let's look into the same



# sex

100*pd.crosstab(train_df.Survived, train_df.Sex, margins = True, margins_name = 'Total', normalize = True).round(4)
# Cross tabulation to see the M and F distribution across different PClass



100*pd.crosstab(train_df.Sex, train_df.Pclass, margins = True, margins_name = "Total", normalize = True).round(3)
100*pd.crosstab(train_df.Pclass,train_df.Survived, normalize = 'index').round(3)
# Survival rate of Siblings/Spouses



100*train_df[["SibSp", 'Survived']].groupby(["SibSp"]).mean().sort_values(by = 'Survived', ascending = False).round(4)
# Survival rate of Parents/Children



100*train_df[["Parch", 'Survived']].groupby(["Parch"]).mean().sort_values(by = 'Survived', ascending = False).round(4)
# Visualizing for age group



train_df['Age_Group'] = pd.cut(train_df.Age, bins = [0,16,32,48,64,100], labels = [0,1,2,3,4,])



plt.figure(figsize = (8,6))

sns.countplot('Age_Group', hue = 'Survived', data= train_df, palette="Set1")



plt.title("Survival distribution according to Age Group", fontsize = 20)

plt.ylabel('Frequency',fontsize = 15)

plt.xlabel('Age Groups', fontsize = 15)

plt.show()
# We will drop the age group column has we dont need it



train_df.drop('Age_Group', inplace = True, axis = 1)
# Visulizing for gender



plt.figure(figsize = (8,6))



sns.countplot('Sex', hue = 'Survived', data= train_df)



plt.title("Survival distribution according to Gender", fontsize = 20)

plt.ylabel('Frequency',fontsize = 15)

plt.xlabel('Gender', fontsize = 15)

plt.show()
# Visualising for different Pclass



g = sns.FacetGrid(train_df, row = 'Pclass', col = 'Survived', height=2.5, aspect=1.5)

g.map(plt.hist, 'Age', alpha = 0.5, bins = 20,edgecolor="black", color = 'g')
# we will fill the missing values in embarked with mode



train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)

test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace = True)
# Visualization for Embrakation



plt.figure(figsize = (8,6))



sns.countplot('Embarked', hue = 'Survived', data = train_df)



plt.title("Survival distribution according to Embrakation", fontsize = 20)

plt.ylabel('Frequency',fontsize = 15)

plt.xlabel('Port of Embarkation', fontsize = 15)

plt.show()





# C = Cherbourg, Q = Queenstown, S = Southampton
100*pd.crosstab(train_df.Embarked, train_df.Survived, normalize = 'index').round(3)
# Let's look into different fare prices for different embarkation port



plt.figure(figsize = (8,6))

sns.barplot(y = 'Embarked', x = 'Fare', data = train_df, hue = 'Pclass', palette = 'Set1', ci = None)



plt.title("Fair prices for various Pclass from different Embrakation port", fontsize = 20)

plt.ylabel('Embrakation Port',fontsize = 15)

plt.yticks([0,1,2], ['Southampton', 'Cherbourg','Queenstown'])

plt.xlabel('Fare Price', fontsize = 15)

plt.show()
# Looking into average fair price according to gender and port embarked`



pd.pivot_table(train_df, index = ['Sex','Embarked'], columns = 'Pclass', values = 'Fare', aggfunc = np.mean).round(2)
#Checking weather duplicate tickets were issued



duplicate = train_df['Ticket'].duplicated().sum()



print("Number of duplicate tickets issued are {0} which contributes around {1}%".format(duplicate, 100*round(duplicate/len(train_df),2) ))
# Dropping Ticket and Passenger ID from data frame as it doesnt contribute for analysis



train_df.drop(['PassengerId', 'Ticket'], inplace = True, axis = 1)



test_df.drop(['PassengerId', 'Ticket'], inplace = True, axis = 1)
# We will replace male to 1 and female to 0 & make sex column numeric



train_df['Sex'].replace(['female', 'male'], [0,1], inplace = True)

test_df['Sex'].replace(['female','male'], [0,1], inplace = True)
# Label encoding embracked column



le = preprocessing.LabelEncoder()



train_df['Embarked'] = le.fit_transform(train_df['Embarked'])

test_df['Embarked'] = le.fit_transform(test_df['Embarked'])
# We will extract a new feature called Title from name



for df in merged:

    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand = False)

    

pd.crosstab(train_df['Title'], train_df['Sex'])
#Replacing the least repeated keywords with others



for df in merged:

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Others')

    

    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    

100*pd.crosstab(train_df.Title, train_df.Survived, normalize = 'index').round(3)
# Encoding Title column



for df in merged:

    df['Title'] = le.fit_transform(df['Title'])
# Dropping the Name column from the dataset



for df in merged:

    df.drop('Name', axis = 1, inplace = True)
# Filling the misssing value for fare in test dataset with median



test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)
# Stroting the column names



train_columns = train_df.columns

test_columns = test_df.columns
# Filling the missing values for age with Iterative Imputer for train



ii = IterativeImputer(initial_strategy='median', min_value = 0, max_value = 80, random_state = 42)



train_df_clean = pd.DataFrame(ii.fit_transform(train_df))

train_df_clean.columns = train_columns
# Similiarly for test



test_df_clean = pd.DataFrame(ii.fit_transform(test_df))

test_df_clean.columns = test_columns
# Restoring the datatype to there original format



main = [train_df_clean, test_df_clean]



for df in main:



    for i in ['Pclass', 'Sex', 'SibSp', 'Parch','Embarked','Title']:

        df[i] = pd.to_numeric(df[i])

        df[i] = df[i].astype(int)
# Changing the datatype of survived in training dataset



train_df_clean['Survived'] = pd.to_numeric(train_df_clean['Survived'])

train_df_clean['Survived'] = train_df_clean['Survived'].astype(int)
train_df_clean.head()
# Creating a new feature called 'Familysize'



for df in main:

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# Family size and surival chances



100 * pd.pivot_table(data = train_df_clean, index = 'FamilySize', values = 'Survived', aggfunc = np.mean).sort_values(by = 'Survived', ascending = False).round(3)
#plotting graph to see surival rate and family size



plt.figure(figsize = (8,6))



sns.lineplot(data = train_df_clean, x = 'FamilySize', y = 'Survived',ci = None, marker="o")



plt.title("Family Size vs Survival Rate", fontsize = 20)

plt.ylabel('Survival Rate',fontsize = 15)

plt.xlabel('Family Size', fontsize = 15)

plt.show()
# Creating another attribute called Is_alone



for df in main:

    df['Is_Alone'] = 0

    df.loc[df['FamilySize']==1, 'Is_Alone'] = 1

    

# 1 = alone & 0 = Not_alone
# Let's look at survival rate of alone passaneger



100 * pd.crosstab(train_df_clean['Is_Alone'], train_df_clean['Survived'], normalize = 'index').round(3)
plt.figure(figsize = (8,6))



sns.barplot(data = train_df_clean, x = 'Is_Alone', y = 'Survived', ci = None)



plt.title("Chances of Solo Passeneger Surviving", fontsize = 20)

plt.ylabel('Survival Rate',fontsize = 15)

plt.xlabel('Type of Passeneger', fontsize = 15)

plt.xticks([0,1], ['Family', 'Solo Passenger'])

plt.show()
plt.figure(figsize = (15,10))



sns.heatmap(train_df_clean.corr(), annot = True)
# based on the above correaltion we will drop SibSp, Parch, Family_size



for df in main:

    df.drop(['SibSp','Parch','FamilySize'], inplace = True, axis =1)
# Plotting box plot for all the variables and checking for outliers





plt.figure



for i, col in enumerate(train_df_clean.columns):

    plt.figure(i)

    sns.boxplot(train_df_clean[col])
# Doing small outlier treatment for fare attribute



train_df_clean.drop(train_df_clean.index[train_df_clean['Fare'] > 300], inplace = True)
# This are the final data frame



train_df_clean.head()
test_df_clean.head()
# Let's look into class imbalance of our target varaible i.e. survived



pd.crosstab(train_df_clean['Survived'], train_df_clean['Survived'], normalize = True).round(4)*100
X_train = train_df_clean.drop('Survived', axis =1)



y_train = train_df_clean['Survived']



X_test = test_df_clean
# Storing the column names  for train and test

X_train_col = X_train.columns



X_test_col = X_test.columns
# We will convert the data into array as it will optimize more



X_train, y_train = np.array(X_train), np.array(y_train)
scaler = MinMaxScaler()



#for train data set

X_train = scaler.fit_transform(X_train)

X_train = pd.DataFrame(X_train, columns = X_train_col)
#Scaling test dataset



X_test = scaler.fit_transform(X_test)

X_test = pd.DataFrame(X_test, columns = X_test_col)
#To use later for random forest



rf_X_train = X_train.copy()

rf_X_test = X_test.copy()
# Finding the optimum hyper paramters



## Different parameters to check

max_iter=[100,110,120,130,140]

C_param_range = [0.001,0.01,0.1,1,10,100]

folds = KFold(n_splits = 5, shuffle = True, random_state = 42)



## Setting the paramters

param_grid = dict(max_iter = max_iter, C = C_param_range)



## Setting model

log = LogisticRegression(penalty = 'l2')



## Set up GridSearch for score metric



grid_search = GridSearchCV(estimator = log, param_grid = param_grid, cv = folds, n_jobs = -1, 

                           return_train_score = True, scoring = 'accuracy')



## Fitting

grid_search.fit(X_train, y_train)
# Looking at the best parameters



print("The best accuracy score is {0:2.3} at {1}".format(grid_search.best_score_, grid_search.best_params_))
# Setting model with optimum parameters



log = LogisticRegression(penalty = 'l2', C = 10, max_iter =100, class_weight = 'balanced')
# Fitting the model



log_fit = log.fit(X_train, y_train)
# Predicting on test data set



y_test_pred = log_fit.predict(X_test)
#Accuracy score for training data



print("Accuracy score for training data is: {0}".format(round(log_fit.score(X_train, y_train) * 100, 2)))
X_train_sm = sm.add_constant(X_train)

log_sm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

res = log_sm.fit()

print(res.summary())
X_train.shape[1]
# Make a VIF dataframe for all the variables present



vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range (X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif.reset_index(drop = True, inplace = True)

vif
# Based on the above values we will drop Title



X_train.drop('Title', axis = 1, inplace = True)
X_train_sm = sm.add_constant(X_train)

log_sm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

res = log_sm.fit()

print(res.summary())
# Make a VIF dataframe for all the variables present



vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range (X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif.reset_index(drop = True, inplace = True)

vif
# Based on the observation we will drop Is_alone



X_train.drop('Is_Alone', axis =1, inplace = True)
X_train_sm = sm.add_constant(X_train)

log_sm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

res = log_sm.fit()

print(res.summary())
# Make a VIF dataframe for all the variables present



vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range (X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif.reset_index(drop = True, inplace = True)

vif
#Dropping Fare



X_train.drop('Fare', inplace = True, axis =1)
X_train_sm = sm.add_constant(X_train)

log_sm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

res = log_sm.fit()

print(res.summary())
# Make a VIF dataframe for all the variables present



vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range (X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif.reset_index(drop = True, inplace = True)

vif
#Prediciting the values of X train



y_train_pred = res.predict(sm.add_constant(X_train))
y_train_pred_final = pd.DataFrame({'Survived': y_train, 'Survived_Proab':y_train_pred})

y_train_pred_final['Survived_Proab'] = round(y_train_pred_final['Survived_Proab'],2)

y_train_pred_final.head(2)
# ROC function



def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
#Storing the values for FPR, TPR and thersolds



fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final['Survived'], y_train_pred_final['Survived_Proab'], drop_intermediate = False )
# Call the ROC function



draw_roc(y_train_pred_final['Survived'], y_train_pred_final['Survived_Proab'])
# Let's create columns with different probability cutoffs 



numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final['Survived_Proab'].map(lambda x: 1 if x > i else 0)

y_train_pred_final.head(2)
# Let's create a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs



cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final['Survived'], y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Plotting sensitivity, accuracy and specificity



sns.set()

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
#Creating final predicated column



y_train_pred_final['final_predicted'] = y_train_pred_final['Survived_Proab'].map( lambda x: 1 if x > 0.4 else 0)



y_train_pred_final.head(2)
#Looking into the accuray of training data set



print("Accuracy : {:2.2}".format(metrics.accuracy_score(y_train_pred_final['Survived'], y_train_pred_final['final_predicted'])))
# Drop the required columns from X_test as well



X_test.drop(['Fare', 'Is_Alone', 'Title'], axis =1, inplace = True)
# Making predictions on test data set



y_test_pred1 = res.predict(sm.add_constant(X_test))
# Converting y_pred to a dataframe



y_pred1 = pd.DataFrame(y_test_pred1, columns = ['Survived_Proab'])

y_pred1.reset_index(drop = True, inplace = True)

y_pred1
# Make predictions on the test set using 0.4 as the cutoff



y_pred1['final_predicted'] = y_pred1['Survived_Proab'].map(lambda x: 1 if x > 0.4 else 0)

y_pred1.head()
#Top features and there importance



IP = pd.DataFrame(res.params , columns = ['Importance'])

IP.reset_index(inplace = True)

IP.columns = ['Features', 'Importance']

IP.drop(IP.index[0], inplace = True)

IP = IP.sort_values(by = 'Importance')

IP.reset_index(drop = True, inplace =True)

IP['Importance'] =  round(IP['Importance'], 2)

IP.head(10)
# Instantiate



rf = RandomForestClassifier()



#Fitting 

rf.fit(rf_X_train, y_train)
# Setting up folds

folds = KFold(n_splits = 5, shuffle = True, random_state = 42)



#Setting up parameters to check

param_grid = {

    'max_depth': [4,8,10],

    'min_samples_leaf': range(100, 400, 200),

    'min_samples_split': range(200, 500, 200),

    'n_estimators': [100,200, 300], 

    'max_features': [4,5,6,7]

}



# Create a based model

rf = RandomForestClassifier()



# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = folds, n_jobs = -1,verbose = 1)
# Fit the grid search to the data



grid_search.fit(rf_X_train, y_train)
# printing the optimal accuracy score and hyperparameters



print('We can get accuracy of',round(grid_search.best_score_,2),'using',grid_search.best_params_)
# model with the best hyperparameters



rf = RandomForestClassifier(bootstrap=True,class_weight = "balanced", criterion = 'gini',

                             max_depth=4,

                             min_samples_leaf=100, 

                             min_samples_split=200,

                             max_features=4,

                             n_estimators=200)

rf_fit = rf.fit(rf_X_train, y_train)
#Predicitng on test



rf_y_pred_test = rf_fit.predict(rf_X_test)
#File submission



submission = pd.DataFrame({"PassengerId": test_valid["PassengerId"], "Survived": rf_y_pred_test})



submission.to_csv('submission_.csv', index=False)