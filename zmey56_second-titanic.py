

# Import our libraries

import pandas as pd

import numpy as np



# Import sklearn libraries

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score

from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc, make_scorer, confusion_matrix, f1_score, fbeta_score



# Import the Naive Bayes, logistic regression, Bagging, RandomForest, AdaBoost, GradientBoost, Decision Trees and SVM Classifier



from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

from xgboost import XGBClassifier



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

#from matplotlib import style

#plt.style.use('bmh')

#plt.style.use('ggplot')

plt.style.use('seaborn-notebook')



from matplotlib.ticker import StrMethodFormatter



from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer

#titanic_features = pd.read_csv('train.csv')

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv("../input/test.csv")
train_df.info()
train_df.shape
test_df.shape
test_df.info()
train_df.head()
train_df.describe()
total = train_df.isnull().sum().sort_values(ascending=False)

percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
train_df.columns.values
from IPython.display import HTML

HTML('''<script>

code_show_err=false; 

function code_toggle_err() {

 if (code_show_err){

 $('div.output_stderr').hide();

 } else {

 $('div.output_stderr').show();

 }

 code_show_err = !code_show_err

} 

$( document ).ready(code_toggle_err);

</script>

To toggle on/off output_stderr, click <a href="javascript:code_toggle_err()">here</a>.''')
train_df['Embarked'].value_counts()/len(train_df)
sns.set(style="darkgrid")

sns.countplot( x='Embarked', data=train_df, hue="Embarked", palette="Set1");
sns.set(style="darkgrid")

sns.countplot( x='Survived', data=train_df, hue="Embarked", palette="Set1");
train_df.groupby('Embarked').mean()
train_df.groupby('Sex').mean()
FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=None, hue_order=None )

FacetGrid.add_legend();
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))

women = train_df[train_df['Sex']=='female']

men = train_df[train_df['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False, color="green")

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False, color="red")

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False, color="green")

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False, color="red")

ax.legend()

_ = ax.set_title('Male');
sns.barplot(x='Pclass', y='Survived', data=train_df);
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=3.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
data = [train_df, test_df]

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'travelled_alone'] = 'No'

    dataset.loc[dataset['relatives'] == 0, 'travelled_alone'] = 'Yes'

    #dataset['travelled_alone'] = dataset['travelled_alone'].astype(int)

train_df['travelled_alone'].value_counts()
test_df['travelled_alone'].value_counts()
train_df['relatives'].value_counts()
axes = sns.factorplot('relatives','Survived', 

                      data=train_df, aspect = 2.5, );
# Drop 'PassengerId' from the train set, because it does not contribute to a persons survival probability.

train_df = train_df.drop(['PassengerId'], axis=1)
train_df['Cabin'].describe()
import re

deck = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "U": "U"}

data = [train_df, test_df]



for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna("U")

    #dataset['Deck'] = dataset['Deck'].astype(int)

# we can now drop the cabin feature

train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)
train_df['Deck'].value_counts()
train_df.groupby('Deck').mean()
test_df['Deck'].value_counts()
data = [train_df, test_df]



for dataset in data:

    mean = train_df["Age"].mean()

    std = test_df["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_df["Age"].astype(int)
train_df["Age"].isnull().sum()
test_df["Age"].isnull().sum()
train_df["Age"].describe()
#train_df.groupby('Age').mean()
train_df['Embarked'].describe()
train_df['Embarked'].mode()
#common_value = train_df['Embarked'].mode()

#common_value
common_value = 'S'

data = [train_df, test_df]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
test_df['Embarked'].describe()
data = [train_df, test_df]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
train_df['Fare'].describe()
train_df['Fare'].isnull().sum()
test_df['Fare'].describe()
test_df['Fare'].isnull().sum()
train_df['Fare'] = train_df['Fare'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)
train_titles = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

type(train_titles)
train_titles.value_counts()
data = [train_df, test_df]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    #dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna("NA")

train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)
train_df.groupby(['Title']).mean()
test_df.groupby(['Title']).mean()
train_df['Sex'].value_counts()
'''

genders = {"male": 0, "female": 1}

data = [train_df, test_df]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)



'''
train_df['Ticket'].describe()
test_df['Ticket'].describe()
train_df = train_df.drop(['Ticket'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1)
'''

ports = {"S": 0, "C": 1, "Q": 2}

data = [train_df, test_df]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)

    

'''    


data = [train_df, test_df]

for dataset in data:

    dataset['Age_Class']= dataset['Age']* dataset['Pclass']

    

    


for dataset in data:

    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)

    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)

    
data = [train_df, test_df]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 7

    

    dataset['Age'] = dataset['Age'].astype(str)

    dataset.loc[ dataset['Age'] == '0', 'Age'] = "Children"

    dataset.loc[ dataset['Age'] == '1', 'Age'] = "Teens"

    dataset.loc[ dataset['Age'] == '2', 'Age'] = "Youngsters"

    dataset.loc[ dataset['Age'] == '3', 'Age'] = "Young Adults"

    dataset.loc[ dataset['Age'] == '4', 'Age'] = "Adults"

    dataset.loc[ dataset['Age'] == '5', 'Age'] = "Middle Age"

    dataset.loc[ dataset['Age'] == '6', 'Age'] = "Senior"

    dataset.loc[ dataset['Age'] == '7', 'Age'] = "Retired"



# let's see how it's distributed 

train_df['Age'].value_counts()
test_df['Age'].value_counts()
train_df.info()
data = [train_df, test_df]



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    dataset['Fare'] = dataset['Fare'].astype(str)

    dataset.loc[ dataset['Fare'] == '0', 'Fare'] = "Extremely Low"

    dataset.loc[ dataset['Fare'] == '1', 'Fare'] = "Very Low"

    dataset.loc[ dataset['Fare'] == '2', 'Fare'] = "Low"

    dataset.loc[ dataset['Fare'] == '3', 'Fare'] = "High"

    dataset.loc[ dataset['Fare'] == '4', 'Fare'] = "Very High"

    dataset.loc[ dataset['Fare'] == '5', 'Fare'] = "Extremely High"

    
train_df['Fare'].value_counts()
test_df['Fare'].value_counts()
train_df.info()
test_df.info()
# Let's take a last look at the training set

train_df.head(10)
train_df.info()
test_df.info()
train_df['Pclass'].value_counts()
data = [train_df, test_df]



for dataset in data:

    dataset['Pclass'] = dataset['Pclass'].astype(str)

    dataset.loc[ dataset['Pclass'] == '1', 'Pclass'] = "Class1"

    dataset.loc[ dataset['Pclass'] == '2', 'Pclass'] = "Class2"

    dataset.loc[ dataset['Pclass'] == '3', 'Pclass'] = "Class3"

    
train_df.info()
test_df.info()
train_df['Pclass'].value_counts()
# Capture all the numerical features so that we can scale them later

#data = [train_df, test_df]

train_numerical_features = list(train_df.select_dtypes(include=['int64', 'float64', 'int32']).columns)

train_numerical_features
type(train_numerical_features)
del train_numerical_features[0]

train_numerical_features
# Feature scaling - Standard scaler

ss_scaler = StandardScaler()

train_df_ss = pd.DataFrame(data = train_df)

train_df_ss[train_numerical_features] = ss_scaler.fit_transform(train_df_ss[train_numerical_features])
train_df_ss.shape
train_df_ss.head()
test_numerical_features = list(test_df.select_dtypes(include=['int64', 'float64', 'int32']).columns)

test_numerical_features
del test_numerical_features[0]

test_numerical_features
# Feature scaling - Standard scaler

test_ss_scaler = StandardScaler()

test_df_ss = pd.DataFrame(data = test_df)

test_df_ss[test_numerical_features] = test_ss_scaler.fit_transform(test_df_ss[test_numerical_features])
test_df.shape
test_df.head()
# One-Hot encoding / Dummy variables

encode_col_list = list(train_df.select_dtypes(include=['object']).columns)

for i in encode_col_list:

    train_df_ss = pd.concat([train_df_ss,pd.get_dummies(train_df_ss[i], prefix=i)],axis=1)

    train_df_ss.drop(i, axis = 1, inplace=True)
train_df_ss.shape
train_df_ss.head()
# One-Hot encoding / Dummy variables

test_encode_col_list = list(test_df.select_dtypes(include=['object']).columns)

for i in test_encode_col_list:

    test_df_ss = pd.concat([test_df_ss,pd.get_dummies(test_df_ss[i], prefix=i)],axis=1)

    test_df_ss.drop(i, axis = 1, inplace=True)
test_df_ss.shape
test_df_ss.head()
X_train = train_df_ss.drop("Survived", axis=1)

Y_train = train_df_ss["Survived"]

X_test  = test_df_ss.drop("PassengerId", axis=1).copy()
X_train.shape
Y_train.shape
X_test.shape
X_train.info()
# Instantiate our model

logreg = LogisticRegression()



# Fit our model to the training data

logreg.fit(X_train, Y_train)



# Predict on the test data

logreg_predictions = logreg.predict(X_test)



logreg_data = pd.read_csv('..//input/test.csv')

logreg_data.insert((logreg_data.shape[1]),'Survived',logreg_predictions)



logreg_data.to_csv('LogisticRegression_SS_OH_FE2.csv')
answer = logreg_data[['PassengerId', 'Survived']]

answer.to_csv('LogisticRegression_two_col.csv', index=False)
answer.head()
# Instantiate our model

adaboost = AdaBoostClassifier()



# Fit our model to the training data

adaboost.fit(X_train, Y_train)



# Predict on the test data

adaboost_predictions = adaboost.predict(X_test)



adaboost_data = pd.read_csv('..//input/test.csv')

adaboost_data.insert((adaboost_data.shape[1]),'Survived',adaboost_predictions)



adaboost_data.to_csv('AdaptiveBoosting_SS_OH_FE.csv')
answer = adaboost_data[['PassengerId', 'Survived']]

answer.to_csv('Adaptive_Boosting.csv', index=False)
# Instantiate our model

bag = BaggingClassifier()



# Fit our model to the training data

bag.fit(X_train, Y_train)



# Predict on the test data

bag_predictions = bag.predict(X_test)



bag_data = pd.read_csv('..//input/test.csv')

bag_data.insert((bag_data.shape[1]),'Survived',bag_predictions)



bag_data.to_csv('Bagging.csv')
answer = bag_data[['PassengerId', 'Survived']]

answer.to_csv('Bagging_Classifier.csv', index=False)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



random_forest_predictions = random_forest.predict(X_test)



rf_data = pd.read_csv('..//input/test.csv')

rf_data.insert((rf_data.shape[1]),'Survived',random_forest_predictions)



rf_data.to_csv('RandomForest_SS_OH.csv')
answer = rf_data[['PassengerId', 'Survived']]

answer.to_csv('Random_Forest.csv', index=False)
# Instantiate our model

dt = DecisionTreeClassifier()

dt.fit(X_train, Y_train)



dt_predictions = dt.predict(X_test)



dt_data = pd.read_csv('..//input/test.csv')

dt_data.insert((dt_data.shape[1]),'Survived',dt_predictions)



dt_data.to_csv('DecisionTrees.csv')
answer = dt_data[['PassengerId', 'Survived']]

answer.to_csv('Decision_Trees.csv', index=False)
# Instantiate our model

gb = GradientBoostingClassifier()

gb.fit(X_train, Y_train)

dt_data

gb_predictions = gb.predict(X_test)



gb_data = pd.read_csv('..//input/test.csv')

gb_data.insert((gb_data.shape[1]),'Survived',gb_predictions)



gb_data.to_csv('GradientBoost_SS_OH_FE.csv')
answer = gb_data[['PassengerId', 'Survived']]

answer.to_csv('Gradient_Boost.csv', index=False)
# Instantiate our model

xg = XGBClassifier(learning_rate=0.02, n_estimators=750,

                   max_depth= 3, min_child_weight= 1, 

                   colsample_bytree= 0.6, gamma= 0.0, 

                   reg_alpha= 0.001, subsample= 0.8

                  )

xg.fit(X_train, Y_train)



xg_predictions = xg.predict(X_test)



xg_data = pd.read_csv('..//input/test.csv')

xg_data.insert((xg_data.shape[1]),'Survived',xg_predictions)



xg_data.to_csv('XGBoost_SS_OH_FE_GSCV.csv')
answer = xg_data[['PassengerId', 'Survived']]

answer.to_csv('XGBoost.csv', index=False)


param_test1 = {

    #'n_estimators': [100,200,500,750,1000],

    #'max_depth': [3,5,7,9],

    #'min_child_weight': [1,3,5],

    'gamma':[i/10.0 for i in range(0,5)],

    'subsample':[i/10.0 for i in range(6,10)],

    'colsample_bytree':[i/10.0 for i in range(6,10)],

    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05, 0.1, 1]

    #'learning_rate': [0.01, 0.02, 0.05, 0.1]

}



scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}



'''

fit_params={"early_stopping_rounds":42, 

            "eval_metric" : "mae", 

            "eval_set" : [[test_features, test_labels]]}

            

'''



gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.02, n_estimators=750,

                   max_depth= 3, min_child_weight= 1), 

                       param_grid = param_test1, #fit_params=fit_params,

                       scoring=scoring, iid=False, cv=3, verbose = 5, refit='Accuracy')

gsearch1.fit(X_train, Y_train)

#gsearch1.grid_scores_, 

gsearch1.best_params_, gsearch1.best_score_
results = gsearch1.cv_results_
results
# Instantiate our model

xg = XGBClassifier(learning_rate=0.02, n_estimators=750,

                   max_depth= 3, min_child_weight= 1, 

                   colsample_bytree= 0.6, gamma= 0.0, 

                   reg_alpha= 0.001, subsample= 0.8

                  )

xg.fit(X_train, Y_train)

xg_predictions = xg.predict(X_test)

xg_data = pd.read_csv('..//input/test.csv')

xg_data.insert((xg_data.shape[1]),'Survived',xg_predictions)

xg_data.to_csv('XGBoost_SS_OH_FE_GSCV.csv')
xg_data.head()
answer = xg_data[['PassengerId', 'Survived']]

answer.to_csv('XGBoost_2.csv', index=False)
test = pd.read_csv("..//input/test.csv")
test['Age'].fillna(test['Age'].median(),inplace=True) # Age

test['Fare'].fillna(test['Fare'].median(),inplace=True) # Fare

d = {1:'1st',2:'2nd',3:'3rd'} #Pclass

test['Pclass'] = test['Pclass'].map(d)

test['Embarked'].fillna(test['Embarked'].value_counts().index[0], inplace=True) # Embarked

ids = test[['PassengerId']]# Passenger Ids

test.drop(['PassengerId','Name','Ticket','Cabin'],1,inplace=True)# Drop Unnecessary Columns

categorical_vars = test[['Pclass','Sex','Embarked']]# Get Dummies of Categorical Variables

dummies = pd.get_dummies(categorical_vars,drop_first=True)

test = test.drop(['Pclass','Sex','Embarked'],axis=1)#Drop the Original Categorical Variables

test = pd.concat([test,dummies],axis=1)#Instead, concat the new dummy variables

test.head()