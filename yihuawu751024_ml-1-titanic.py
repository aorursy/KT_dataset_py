#load packages
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
#ignore warnings
import warnings
warnings.filterwarnings('ignore')
#import data from file
train= pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
data_clean=[train,test]

#preview data
print(train.info())
train.sample(5)
##NULL VALUE 
#count the null values columns
print('Train:Columns with null value:\n', train.isnull().sum())
print('-'*10)
print('Test:Columns with null value:\n', test.isnull().sum())
#Clean or Imputation for null values
for dataset in data_clean:
  dataset['Age'].fillna(dataset['Age'].median(),inplace=True)
  dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
  dataset['Fare'].fillna(dataset['Fare'].median(),inplace=True)

#Drop columns will not be used for training
drop_column=['PassengerId','Name','Cabin','Ticket']
train.drop(drop_column, axis=1, inplace= True)

#View the data after cleanning
print(train.isnull().sum())
print('-'*10)
print(train.isnull().sum())
#Convert object to category using Label Encoder 
print(train.info())
from sklearn import preprocessing
le= preprocessing.LabelEncoder()

for dataset in data_clean:
    dataset['Embarked_Code']=le.fit_transform(dataset['Embarked'])
    dataset['Sex_Code']=le.fit_transform(dataset['Sex'])
    
#Define y variable as target
Target=['Survived']

#Define x variables##remove SibSp based on the correlation
train_x=['Sex','Pclass','Embarked','Parch','Age','Fare']
train_x_cal=['Sex_Code','Pclass','Embarked_Code','Parch','Age','Fare']
#Review correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    
    _ = sns.heatmap(
        df.corr(), 
        center=0,
        ax=ax,
        linewidths=0.1,
        annot=True, 
        annot_kws={'fontsize':14 }
    )
    plt.title('Correlation of Features', y=1.05, size=15)

correlation_heatmap(train)
#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    #xgboost: 
    XGBClassifier()    
    ]

#split dataset in cross-validation with this splitter class
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = train[Target]

#index through MLA and save performance to table
row_index = 1
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    
    #score model with cross validation
    cv_results = model_selection.cross_validate(alg, train[train_x_cal], train[Target], cv  = cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    #save MLA predictions - see section 6 for usage
    alg.fit(train[train_x_cal], train[Target])
    MLA_predict[MLA_name] = alg.predict(train[train_x_cal])
    
    row_index+=1

    
#print and sort table
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare


print(test.info())
#tune each estimator before creating a super model
grid_n_estimator = [50,100,300]
grid_ratio = [.1,.25,.5,.75,1.0]
grid_learn = [.01,.03,.05,.1,.25]
grid_max_depth = [2,4,6,None]
grid_min_samples = [5,10,.03,.05,.10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

#gradient boosting w/full dataset modeling submission score
submit_gbc = ensemble.GradientBoostingClassifier()
submit_gbc = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid={'learning_rate': grid_ratio, 'n_estimators': grid_n_estimator, 'max_depth': grid_max_depth, 'random_state':grid_seed}, scoring = 'roc_auc', cv = cv_split)
submit_gbc.fit(train[train_x_cal], train[Target])
print('Best Parameters: ', submit_gbc.best_params_) 
test['Survived'] = submit_gbc.predict(test[train_x_cal])

#submit file
submit = test[['PassengerId','Survived']]
submit.to_csv("submit.csv", index=False)

print('Validation Data Distribution: \n', test['Survived'].value_counts(normalize = True))
submit.sample(10)
#Plot the probability of survived based on Sex, Pclass and Embarked(most correlated features)
#Found the the probability of survival distribution:Female>Male, Pclass1>2>3, Embarked C>Q>S
fig, (axis1,axis2,axis3) = plt.subplots(1, 3,figsize=(22,5))
sns.barplot(x = 'Sex', y = 'Survived', order=['female','male'], data=train,palette="Set2", ax = axis1)
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=train,palette="Set2", ax = axis2)
sns.barplot(x = 'Embarked', y = 'Survived', order=['C','Q','S'],data=train,palette="Set2", ax = axis3)
#Verify whether the low probability of survival at Pclass 3 is because of passenger density
sns.countplot('Pclass', data=train,palette="Set2")
#Verify whether the high probability of survival within Female is because most of them are at Pclass 1 and 2.-->No
h = sns.FacetGrid(train, col = 'Sex', hue = 'Survived')
h.map(sns.countplot, 'Pclass', alpha = .7)
h.add_legend()

#Verify whether the low probability of survival within Embarked S is because most of them are at Pclass 3.-->Yes
h = sns.FacetGrid(train, col = 'Embarked', hue = 'Survived')
h.map(sns.countplot, 'Pclass', alpha = .7)
h.add_legend()
