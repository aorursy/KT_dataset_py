#################################################

#### import our desired libraries

#################################################



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn, sklearn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier



# setup a style to view ipython notebook graphs

sns.set_style('whitegrid')



#### import the data

test   = pd.read_csv('../input/test.csv')

train    = pd.read_csv('../input/train.csv')
#################################################

#### feature engineering 

#################################################





###train['Ticket_group'] = np.where(train['Ticket'].str.isdigit(), train['Ticket'].astype(str).str[0], train['Ticket'].str[:1])

train['Ticket_length'] = train['Ticket'].apply(lambda x: len(x))

###test['Ticket_group'] = np.where(test['Ticket'].str.isdigit(), test['Ticket'].astype(str).str[0], test['Ticket'].str[:1])

test['Ticket_length'] = test['Ticket'].apply(lambda x: len(x))



train["NameLength"] = train["Name"].apply(lambda x: len(x))

test["NameLength"] = test["Name"].apply(lambda x: len(x))







########## this counts the number of spaces in the Name column

import re



at = re.compile(r" ", re.I)

def count_spaces(string):

    count = 0

    for i in at.finditer(string):

        count += 1

    return count



train["spaces_in_name"] = train["Name"].map(count_spaces)

test["spaces_in_name"] = test["Name"].map(count_spaces)





#########################################################

##### This function returns the title from a name

#########################################################



def title(name):

# Search for a title using a regular expression. Titles are made of capital and lowercase letters ending with a period.

    find_title = re.search(' ([A-Za-z]+)\.', name)

# Extract and return the title If it exists. 

    if find_title:

        return find_title.group(1)

    return ""



train["Title"] = train["Name"].apply(title)

test["Title"] = test["Name"].apply(title)



##### making some edits for the very small title groups 

train['Title'] = train['Title'].replace(['Don', 'Capt', 'Major', 'Sir', 'Rev', 'Col'], 'Sir')

train['Title'] = train['Title'].replace(['Dona', 'Lady', 'the Countess', 'Jonkheer', 'Mme', 'Mlle', 'Countess'], 'Lady')

train['Title'] = train['Title'].replace(['Ms'], 'Miss')



test['Title'] = test['Title'].replace(['Don', 'Capt', 'Major', 'Sir', 'Rev', 'Col'], 'Sir')

test['Title'] = test['Title'].replace(['Dona', 'Lady', 'the Countess', 'Jonkheer', 'Mme', 'Mlle', 'Countess'], 'Lady')

test['Title'] = test['Title'].replace(['Ms'], 'Miss')







#########################################################

##### add some additional interesting vars

#########################################################



train['Cabin_first_ltr'] = np.where(train['Cabin'].isnull(), 'Null', 'Not Null')

##train['Parch_grouped'] = np.where(train['Parch'] > 0, '1', '0')

train['FamilySize'] = train['SibSp'] + train['Parch']

train['withfamily'] = np.where(train['FamilySize'] > 0, 1, 0)

train['Female'] = np.where(train['Sex'] == 'female', 1, 0)



train['miss'] = np.where(train['Name'].str.contains("Miss. "), 1, 0)

train['mrs'] = np.where(train['Name'].str.contains("Mrs. "), 1, 0)







#########################################################

##### Group up the ticket variable

#########################################################

train["Ticket_grp"] = np.where(train['Ticket'].str.isdigit(), train['Ticket'].astype(str).str[0], train["Ticket"].str.split(' ').str.get(0))

test["Ticket_grp"] = np.where(test['Ticket'].str.isdigit(), test['Ticket'].astype(str).str[0], test["Ticket"].str.split(' ').str.get(0))

###########################################################

#### bucket the continuous age variable into categories

###########################################################



train['Age_grouped'], bins = pd.qcut(train['Age'], 10, retbins = True)

test['Age_grouped'] = pd.cut(test["Age"], bins=bins, include_lowest=True)



train['Fare_grouped'], Fare_bins = pd.qcut(train['Fare'], 4, retbins = True)

test['Fare_grouped'] = pd.cut(test["Fare"], bins=Fare_bins, include_lowest=True)





###### note that some of the age values were missing and thus the age_grouped is missing too

#train['Age_grouped'][train['Age'] == train['Age'].median()].unique()



train['Age_grouped'] = np.where(train['Age_grouped'].isnull(), "(25, 28]", train['Age_grouped'])

test['Age_grouped'] = np.where(test['Age_grouped'].isnull(), "(25, 28]", test['Age_grouped'])



train['Fare_grouped'] = np.where(train['Fare_grouped'].isnull(), "(7.91, 14.454]", train['Fare_grouped'])

test['Fare_grouped'] = np.where(test['Fare_grouped'].isnull(), "(7.91, 14.454]", test['Fare_grouped'])



##train['Fare_grouped'][train['Fare'] == train['Fare'].median()].unique()
### what's the distribution of age? 

#sns.violinplot(data=train['Age'])



#sns.violinplot(data = train['Fare'])
## to run a random forest we need to make sure the dataset doens't contain any missing values.

### does it contain missing values

if train.isnull().values.any() == True:

    print("there are some missing values")

else: 

    print("there are no missing values")
## Make adjustments to the test dataset to match the train dataset



test['Cabin_first_ltr'] = np.where(test['Cabin'].isnull(), 'Null', 'Not Null')

test['FamilySize'] = test['SibSp'] + test['Parch']

test['withfamily'] = np.where(test['FamilySize'] > 0, 1, 0)

test['Female'] = np.where(test['Sex'] == 'female', 1, 0)



test['miss'] = np.where(test['Name'].str.contains("Miss. "), 1, 0)

test['mrs'] = np.where(test['Name'].str.contains("Mrs. "), 1, 0)
##### this removes the missing values

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],

            index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.fill)

    





### this will transfer the categorical variables to floats for the algo

def do_treatment(df):

    for col in df:

        if df[col].dtype == np.dtype('O'):

            df[col] = df[col].apply(lambda x : hash(str(x)))



    


train_imputed = DataFrameImputer().fit_transform(train)

test_imputed = DataFrameImputer().fit_transform(test)



do_treatment(train_imputed)

do_treatment(test_imputed)
######## Creating the random forest model 

# Create the random forest object which will include all the parameters

# for the fit

forest = RandomForestClassifier(n_estimators = 200, max_features = 'sqrt',

                             max_depth = None, verbose = 1, n_jobs = -1)



# Fit the training data to the Survived labels and create the decision trees

#train_independent_vars = train_imputed.drop(['Survived'], axis = 1)

train_independent_vars = train_imputed[['Ticket_length', 'Title', 'NameLength', 'Pclass', 'Female', 'Age_grouped', 'Ticket_grp', 'Fare_grouped', 'Cabin_first_ltr', 'spaces_in_name']]

train_independent_vars = train_independent_vars

##, 'Embarked', 'withfamily'

train_dependent_vars = train_imputed['Survived']



forest = forest.fit(train_independent_vars, train_dependent_vars)



# Take the same decision trees and run it on the test data

output = forest.predict(test_imputed[['Ticket_length', 'Title', 'NameLength', 'Pclass', 'Female', 'Age_grouped', 'Ticket_grp', 'Fare_grouped', 'Cabin_first_ltr', 'spaces_in_name']])



### combine the passengerid with the prediction

output_df = pd.DataFrame(test_imputed.PassengerId).join(pd.DataFrame(output))

output_df.columns = ['PassengerId', 'Survived']

#### create the final output dataframe

final_output = DataFrame(columns=['PassengerId', 'Survived'])

final_output = final_output.append(output_df[['PassengerId', 'Survived']])



#### convert from string to ints 

final_output['PassengerId'] = final_output['PassengerId'].astype(int)

final_output['Survived'] = final_output['Survived'].astype(int)



#### convert to csv

final_output.to_csv('output.csv', index = False, header = ['PassengerId', 'Survived'])
#

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in indices:

    print(train_independent_vars.columns[f], importances[f])
#
#
#
#
#