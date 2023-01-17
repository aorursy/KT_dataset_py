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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from time import time


data_raw = pd.read_csv( '/kaggle/input/titanic/train.csv', index_col='PassengerId')
data_validate = pd.read_csv( '/kaggle/input/titanic/test.csv', index_col='PassengerId')
# data_raw = pd.read_csv("datasets/titanic_train.csv", index_col='PassengerId')
# data_validate = pd.read_csv("datasets/titanic_test.csv", index_col='PassengerId')
data_raw.sample(10)
data_raw.info()
data_raw.isnull().sum()
data_raw.describe(include='all')
data_raw['Sex'].value_counts()
data_raw['Embarked'].value_counts()
#Cleaning and Wrangling the Data
data_copy = data_raw.copy(deep=True)
data_cleaner = [data_copy, data_validate]
for dataset in data_cleaner:
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    dataset.drop(['Cabin', 'Ticket', 'Fare', 'Name'], axis=1, inplace = True)

for dataset in data_cleaner:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # We set IsAlone to 1/True for everyone and then change it to 0/False depending on their FamilySize.
    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
    dataset.drop(['SibSp', 'Parch'], axis=1, inplace = True)
data_cleaner[0].head()
for dataset in data_cleaner:
    dataset['Sex'].loc[dataset['Sex'] == 'male'] = 0
    dataset['Sex'].loc[dataset['Sex'] == 'female'] = 1
    dataset['Embarked'].loc[dataset['Embarked'] == 'C'] = 0
    dataset['Embarked'].loc[dataset['Embarked'] == 'Q'] = 1
    dataset['Embarked'].loc[dataset['Embarked'] == 'S'] = 2

data_cleaner[0].head()
##setting data
data_clean, data_validate = data_cleaner
data_labels = data_clean['Survived']
data_features = data_clean.drop('Survived', axis=1)

features_train, features_test, labels_train, labels_test = train_test_split(data_features, data_labels,
                                                                            test_size=0.2, random_state=42)
#training data
features_train.head()
labels_train.head()
## Validation Data
data_validate.head()
    # 2.4.3 Dicision Tree
dt_classifier = tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()
dt_classifier.fit(features_train, labels_train)
print("Training Time: ", round(time() - t0), "s")
t1 = time()
dt_prediction = dt_classifier.predict(features_test)
print("Prediction Time: ", round(time() - t1), "s")
print(accuracy_score(labels_test, dt_prediction))
features_test.head()
dt_classifier.predict(features_test.head())
labels_test[:5]
final = dt_classifier.predict(data_validate)

sample = pd.read_csv("/kaggle/input/titanic/test.csv", index_col='PassengerId')
sample['Survived'] = final
sample.to_csv("titanic_output.csv", )
    # 2.4.2 Naïve Bayes
nb_classifier = GaussianNB()

t0 = time()
nb_classifier.fit(features_train, labels_train)
print("Training Time: ", time()-t0, "s.", sep='')

t1 = time()
nb_pred = nb_classifier.predict(features_test)
print("Testing Time: ", time()-t1, "s.", sep='')
print("Accuracy: ", accuracy_score(labels_test, nb_pred), ".", sep='')
    # 2.4.3 Neural Network
    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# ignore Deprecation Warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Neural Network
import keras 
from keras.models import Sequential 
from keras.layers import Dense
train_df = pd.read_csv( '/kaggle/input/titanic/train.csv')
test_df = pd.read_csv( '/kaggle/input/titanic/test.csv')    
df = train_df.append(test_df , ignore_index = True)   
df['Title'] = df.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip()) 
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace(['Mme','Lady','Ms'], 'Mrs')
df.Title.loc[ (df.Title !=  'Master') & (df.Title !=  'Mr') & (df.Title !=  'Miss') 
             & (df.Title !=  'Mrs')] = 'Others'
df = pd.concat([df, pd.get_dummies(df['Title'])], axis=1).drop(labels=['Name'], axis=1)
# map the two genders to 0 and 1
df.Sex = df.Sex.map({'male':0, 'female':1})

# create a new feature "Family"
df['Family'] = df['SibSp'] + df['Parch'] + 1
df.Family = df.Family.map(lambda x: 0 if x > 4 else x)

df.Ticket = df.Ticket.map(lambda x: x[0])

# inspect the correlation between Ticket and Survived
df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
df[['Ticket', 'Fare']].groupby(['Ticket'], as_index=False).mean()
df[['Ticket', 'Pclass']].groupby(['Ticket'], as_index=False).mean()
# check if there is any NAN
df.Fare.isnull().sum(axis=0)
df.Ticket[df.Fare.isnull()]
df.Pclass[df.Fare.isnull()]
df.Cabin[df.Fare.isnull()]
df.Embarked[df.Fare.isnull()]
guess_Fare = df.Fare.loc[ (df.Ticket == '3') & (df.Pclass == 3) & (df.Embarked == 'S')].median()
df.Fare.fillna(guess_Fare , inplace=True)

# inspect the mean Fare values for people who died and survived
df[['Fare', 'Survived']].groupby(['Survived'],as_index=False).mean()
# bin Fare into five intervals with equal amount of people
df['Fare-bin'] = pd.qcut(df.Fare,5,labels=[1,2,3,4,5]).astype(int)

# inspect the correlation between Fare-bin and Survived
df[['Fare-bin', 'Survived']].groupby(['Fare-bin'], as_index=False).mean()
df = df.drop(labels=['Cabin'], axis=1)
df.describe(include=['O']) # S is the most common
# fill the NAN
df.Embarked.fillna('S' , inplace=True )
df = df.drop(labels='Embarked', axis=1)
import numpy as np
# notice that instead of using Title, we should use its corresponding dummy variables 
df_sub = df[['Age','Master','Miss','Mr','Mrs','Others','Fare-bin','SibSp']]

X_train  = df_sub.dropna().drop('Age', axis=1)
y_train  = df['Age'].dropna()
X_test = df_sub.loc[np.isnan(df.Age)].drop('Age', axis=1)

regressor = RandomForestRegressor(n_estimators = 300)
regressor.fit(X_train, y_train)
y_pred = np.round(regressor.predict(X_test),1)
df.Age.loc[df.Age.isnull()] = y_pred

df.Age.isnull().sum(axis=0) # no more NAN now

bins = [ 0, 4, 12, 18, 30, 50, 65, 100] # This is somewhat arbitrary
age_index = (1,2,3,4,5,6,7) #('baby','child','teenager','young','mid-age','over-50','senior')
df['Age-bin'] = pd.cut(df.Age, bins, labels=age_index).astype(int)

df[['Age-bin', 'Survived']].groupby(['Age-bin'],as_index=False).mean()

df['Ticket'] = df['Ticket'].replace(['A','W','F','L','5','6','7','8','9'], '4')

# check the correlation again
df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
df = pd.get_dummies(df,columns=['Ticket'])

df.head()

df = df.drop(labels=['SibSp','Parch','Age','Fare','Title'], axis=1)
y_train = df[0:891]['Survived'].values
X_train = df[0:891].drop(['Survived','PassengerId'], axis=1).values
X_test  = df[891:].drop(['Survived','PassengerId'], axis=1).values

# Initialising the NN
model = Sequential()

# layers
model.add(Dense(9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))
model.add(Dense(9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# summary
model.summary()
# Compiling the NN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the NN
model.fit(X_train, y_train, batch_size = 32, epochs = 100)
y_pred = model.predict(X_test)
y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])

output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_final})
output.to_csv('prediction.csv', index=False)
y_final
from sklearn.model_selection import cross_validate
from sklearn.datasets import  load_iris
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}
scores_log = cross_validate(logreg,features_train, labels_train, scoring=scoring,
                         cv=5, return_train_score=True)  
print('Scores each Fold : ')
scores_log
print('Test Score','\n','test_accuracy: ', scores_log['test_acc'].mean(),'**','\n',
      'test_precision: ', scores_log['test_prec_macro'].mean(),'\n',
      'test_recall: ', scores_log['test_rec_micro'].mean(),'\n',
      'test_f1_score: ',  2* (((scores_log['test_prec_macro'].mean())*(scores_log['test_rec_micro'].mean())) / ((scores_log['test_prec_macro'].mean())+(scores_log['test_rec_micro'].mean()))),'***','\n',
      '----------------------------------------------------','\n',
      'Train Score','\n','train_accuracy: ', scores_log['test_acc'].mean(),'**','\n',
      'train_precision: ', scores_log['train_prec_macro'].mean(),'\n',
      'train_recall: ', scores_log['train_rec_micro'].mean(),'\n',
      'train_f1_scorel: ',2* (((scores_log['train_prec_macro'].mean())*(scores_log['train_rec_micro'].mean())) / ((scores_log['train_prec_macro'].mean())+(scores_log['train_rec_micro'].mean()))),'***','\n',
      'score_time: ', scores_log['score_time'].mean(),'\n',
      'fit_time: ', scores_log['fit_time'].mean())
#Train กับชุดtrain อีกรอบ
clf_log= logreg.fit(features_train, labels_train)
#ทำนายชุด Test
y_pred_log = clf_log.predict(features_test)

y_pred_log
# Random Forest
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(features_train, labels_train)
Y_prediction = random_forest.predict(features_test)

random_forest.score(features_train, labels_train)

acc_random_forest = round(random_forest.score(features_train, labels_train) * 100, 2)
print(round(acc_random_forest,2,), "%")
#วัดประสิทธิภาพ
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(labels_test, Y_prediction))
print(classification_report(labels_test, Y_prediction))
