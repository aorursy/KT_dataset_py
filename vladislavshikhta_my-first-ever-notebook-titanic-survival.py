import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
import scipy.stats
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from pprint import pprint

%matplotlib inline
# downloading and observing the data
train = pd.read_csv('../input/titanic/train.csv')
train.head()
test = pd.read_csv('../input/titanic/test.csv')
test.head()
y = train.Survived
Id = test.PassengerId
# distribution of survived and died passengers
fig = plt.figure(figsize = (7, 7))
sns.set(style='darkgrid')
ax = sns.countplot(x='Survived', data = train)
print('Number of people who survived: {}'.format(y.value_counts()[1]))
print('Number of people who died: {}'.format(y.value_counts()[0]))
print('Percentage of people who survived: {:.2f}%'.format(y.value_counts(normalize=True)[1]*100))
print('Percentage of people who died: {:.2f}%'.format(y.value_counts(normalize=True)[0]*100))
fig = plt.figure(figsize = (7, 7))
sns.set(style='darkgrid')
ax = sns.countplot(x='Sex', data = train)
print('Number of males: {}'.format(train['Sex'].value_counts()[1]))
print('Number of females: {}'.format(train['Sex'].value_counts()[0]))
sns.catplot(x="Sex",col="Survived",
                data=train, kind="count",
                height=6, aspect=.9)

# print percentages of females vs. males that survive
print("Percentage of females who survived: {:.2f}%".format (train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100))
print("Percentage of males who survived: {:.2f}%". format(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100))
# females are much more likely to survive than male 
fig = plt.figure(figsize = (7, 7))
ax = sns.countplot(x='Pclass', data=train)
sns.catplot(x="Pclass",col="Survived",
                data=train, kind="count",
                height=7, aspect=.7)


print("Percentage of Pclass = 1 who survived: {:.2f}%".format (train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100))
print("Percentage of Pclass = 2 who survived: {:.2f}%".format (train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100))
print("Percentage of Pclass = 3 who survived: {:.2f}%".format (train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100))
# we can see that 1 class are likely to survive than other classes which make sence
# Intuition: combine columns 'SibSp' and 'Parch' to a new column 'family'
# (according to the description of the dataset) and look if family helps to survive or not
train['family'] = train['SibSp'] + train['Parch']
train['family'] = [0 if x == 0 else 1 for x in train['family']]
sns.catplot(x="family",col="Survived",
                data=train, kind="count",
                height=6, aspect=.9)
print("Percentage of persons without family who survived: {:.2f}%".format (train["Survived"][train["family"] == 0].value_counts(normalize = True)[1]*100))
print("Percentage of persons with family who survived: {:.2f}%". format(train["Survived"][train["family"] == 1].value_counts(normalize = True)[1]*100))
# this make sence
# let's see how 'age' influence on survival rate
train.groupby('Survived').Age.value_counts().plot(kind='hist')
sns.jointplot(x = train['Age'], y=y)
train['Survived'][train['Age'] < 10.0].value_counts().plot(kind='barh')
print('Percentage of people with Age < 10 who survived: {:.2f}%'.format(train['Survived'][train['Age'] < 10.0].value_counts(normalize = True)[1]*100))
train['Survived'][(train['Age'] >= 10.0) & (train['Age'] < 25.0)].value_counts().plot(kind='barh')
print('Percentage of people with Age >= 10 and Age < 25 who survived: {:.2f}%'.format(train['Survived'][(train['Age'] >= 10.0) & (train['Age'] < 25.0)].value_counts(normalize = True)[1]*100))
train['Survived'][train['Age'] >= 45.0].value_counts().plot(kind='barh')
print('Percentage of people with Age >= 45 who survived: {:.2f}%'.format(train['Survived'][train['Age'] >= 45.0].value_counts(normalize = True)[1]*100))
# It's controversial question to remove this feature or not. Let's make iteractions after dealing with NaN and than decide
# removing 'Survived' and 'family' columns
train.drop(['Survived','family'], axis = 1, inplace = True)
train.head()
# combining test and training data for cleaning dataset
dataset = pd.concat([train, test], axis = 0 , sort = False, ignore_index = True)
dataset.head()
# Intuition: Name, Ticket, PassengerId and Embarked are not informative features for prediction. So we can merely remove them from dataset
dataset.drop(['PassengerId', 'Ticket', 'Name', 'Embarked'], axis = 1, inplace = True)
dataset['Family'] = dataset['SibSp'] + dataset['Parch']
dataset['Family'] = [0 if x == 0 else 1 for x in dataset['Family']]
dataset.drop(['SibSp', 'Parch'], axis = 1, inplace = True)
dataset.head()
# It's time to handle with missing data
dataset.isna().sum()
print('Percentage of missing values of "Age" column : {:.2f}%'.format(dataset['Age'].isna().mean()*100))
print('Percentage of missing values of "Cabin" column : {:.2f}%'.format(dataset['Cabin'].isna().mean()*100))
print('Percentage of missing values of "Fare" column : {:.2f}%'.format(dataset['Fare'].isna().mean()*100))
# 'Cabin' column has almost all missing values, so we have to delete this feature
dataset.drop('Cabin', axis = 1, inplace = True)
dataset.columns
# Let's look at 'Age' distributions and decide what to do with missing values
# visualizing distributions
sns.distplot(dataset['Age'], kde = False, hist = True)
ax = sns.boxplot(x=dataset['Age'])
# Well, it seems that it is not a good idea to replace all NaN values in the 'Age' column with mean/median/mode
# Let's create a knn model to predict Age
# but before that I will create dummies for 'Sex' column
dataset = pd.get_dummies(dataset, columns = ['Sex'])
dataset = dataset.drop(['Sex_male'], axis = 1)
dataset.head()
# function for KNN model-based imputation of missing values using features without NaN as predictors

def impute_model(dataset):
    cols_nan = dataset.columns[dataset.isna().any()].tolist()    
    cols_no_nan = dataset.columns.difference(cols_nan).values            
    for col in cols_nan:
        test_data = dataset[dataset[col].isna()]
        train_data = dataset.dropna()
        knr = KNeighborsRegressor(n_neighbors=5).fit(train_data[cols_no_nan], train_data[col])
        dataset.loc[dataset[col].isna(), col] = knr.predict(test_data[cols_no_nan])
    return dataset
dataset = impute_model(dataset)
dataset.head()
dataset.isna().sum()
# Great! It's time to deal with 'Pclass' and 'Fare' column
# Intuition: The higher price of your ticket the better class
# Let's check if 'Pclass' and 'Fare' highly correlated
# 0-hypothesis: 'Pclass' and 'Fare' are not correlated
scipy.stats.pearsonr(dataset['Pclass'], dataset['Fare'])    # Pearson's r
scipy.stats.spearmanr(dataset['Pclass'], dataset['Fare'])    # Spearman's rho
scipy.stats.kendalltau(dataset['Pclass'], dataset['Fare'])   # Kendall's tau
# pvalue < 0.05 0-hypothesis is rejected -  these two features are highly correlated
# We will remove 'Fare' 
dataset.drop('Fare', axis = 1, inplace = True)
dataset.head()
# Now I will make a strange step
# As we have already convinced first class are likely to survive than third
# So I will do manual feature scaling
dataset = dataset.replace({'Pclass':{3:0, 2:0.5}}) # I suppouse it's better for ML algorithm
dataset.head()
dataset['ScaleredAge'] = [(x - min(dataset['Age'])) / (max(dataset['Age']) - min(dataset['Age'])) for x in dataset['Age']]
dataset.drop(['Age'], axis = 1, inplace = True)
dataset.head()
# making feature iteractions
def add_iteractions(dataset):
    # Get feature names
    combos = list(combinations(list(dataset.columns), 2))
    colnames = list(dataset.columns) + ['_'.join(x) for x in combos]
    
    # Find interactions
    poly = PolynomialFeatures(interaction_only = True, include_bias = False)
    dataset = poly.fit_transform(dataset)
    dataset = pd.DataFrame(dataset)
    dataset.columns = colnames
    
    # Remove interactions terms with 0 values
    noint_indices = [i for i, x in enumerate(list((dataset == 0).all())) if x]
    dataset = dataset.drop(dataset.columns[noint_indices], axis = 1)
    
    return dataset
dataset = add_iteractions(dataset)
dataset.head()
train = dataset[:len(train)]
train.shape
test = dataset[len(train):]
test.shape
# Let's combine 'y' and 'train' to see correlation map
train_data = pd.concat([y, train], axis = 1, sort = False)
#get correlations of each features in dataset
corrmat = train_data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(train_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# Features to delete: 'ScaleredAge', 'Family_ScaleredAge'
train.drop(['ScaleredAge', 'Family_ScaleredAge'], axis = 1, inplace = True)
test.drop(['ScaleredAge', 'Family_ScaleredAge'], axis = 1, inplace = True)
# That's all for feature engineering
# Models to build and compare : K-NN, Decision Trees, Random Forest, Logistic Regression and ComplementNB
# K-NN
knn = KNeighborsClassifier()
grid_params = {
    'n_neighbors': list(range(1,16)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
cv = StratifiedKFold(n_splits=10, shuffle = True, random_state = 42) # best for classification
gs_knn = GridSearchCV(knn,grid_params, cv = cv, verbose=1, n_jobs = -1)
gs_knn.fit(train, y)
gs_knn.best_score_
def plot_search_results(grid):
    """
    Params: 
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = gs_knn.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    
    

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(gs_knn.best_params_.keys())
    for p_k, p_v in gs_knn.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params=gs_knn.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])

        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')

        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()
plot_search_results(gs_knn)
best_knn = gs_knn.best_estimator_
final_predictions_knn = best_knn.predict(test)
output = pd.DataFrame({'PassengerId': Id, 'Survived': final_predictions_knn})
output.to_csv('submission1.csv', index=False)
# Let's download submission to Kaggle and see the result
print('Kaggle K-NN score : 0.74880')
# Decision Trees
dt = DecisionTreeClassifier()
grid_params_tree = {
    'min_samples_split' : [2, 3, 4, 5, 6],
    'max_leaf_nodes': list(range(2, 100)),
    'max_depth': list(range(1,20,2))
}
gs_dt = GridSearchCV(dt,grid_params_tree, cv = cv, verbose=1, n_jobs = -1)
gs_dt.fit(train, y)
gs_dt.best_score_
best_dt = gs_dt.best_estimator_
from sklearn import tree
tree.plot_tree(best_dt)
final_prediction_dt = best_dt.predict(test)
output = pd.DataFrame({'PassengerId': Id, 'Survived': final_prediction_dt})
output.to_csv('submission2.csv', index=False)
print('Kaggle DT score : 0.76555')
# Random Forest
rf = RandomForestRegressor()
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())
param_grid = {
    'bootstrap': [True],
    'max_depth': [100, 105, 110, 115, 120],
    'max_features': ['sqrt'],
    'min_samples_leaf': [5, 6, 7],
    'min_samples_split': [4, 5, 6],
    'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 300, num = 5)]
}

# Instantiate the grid search model
grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = cv, n_jobs = -1, verbose = 1)
grid_search_rf.fit(train, y)
grid_search_rf.best_params_
best_grid_rf = grid_search_rf.best_estimator_
final_predictions = best_grid_rf.predict(test)
print(final_predictions)
final_predictions[final_predictions >= 0.5] = 1
final_predictions[final_predictions < 0.5] = 0
final_predictions = final_predictions.astype(int)
print(final_predictions[0:5])
output = pd.DataFrame({'PassengerId': Id, 'Survived': final_predictions})
output.to_csv('submission3.csv', index=False)
print('Kaggle Random Forest score : 0.77033')
# Logistic regression
grid_lr = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid_lr,cv=cv, n_jobs = -1, verbose = 1)
logreg_cv.fit(train, y)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
best_score_lr = logreg_cv.best_estimator_
predictions_lr = best_score_lr.predict(test)
output = pd.DataFrame({'PassengerId': Id, 'Survived': predictions_lr})
output.to_csv('submission4.csv', index=False)
print('Kaggle Logistic Regression score : 0.77751')
# Complement NB
NB = ComplementNB()
params_NB = {'alpha':list(np.arange(0.1,10,0.5))}
grid_search_NB = GridSearchCV(estimator = NB, param_grid = params_NB, 
                          cv = cv, n_jobs = -1, verbose = 1)
grid_search_NB.fit(train, y)
best_score_NB = grid_search_NB.best_estimator_
predictions_NB = best_score_NB.predict(test)
output = pd.DataFrame({'PassengerId': Id, 'Survived': predictions_NB})
output.to_csv('submission5.csv', index=False)
print('Kaggle Complemet NB score : 0.76555')
# Let's summarize all results
print('K-NN score : 0.74880')
print('DT score : 0.76555')
print('Random Forest score : 0.77033')
print('Logistic Regression score : 0.77751')
print('Complemet NB score : 0.76555')
