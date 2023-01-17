import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display
%matplotlib inline

train_input = pd.read_csv("../input/train.csv", dtype={"Age": np.float64})
test_input = pd.read_csv("../input/test.csv", dtype={"Age": np.float64})

df = pd.concat([train_input, test_input], ignore_index=True)
df.head()
print(df.hist())
categorical_columns = ['Sex', 'Embarked']
numerical_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
text_columns = ['Name', 'Ticket']

def category_to_numeric(df, column_name):
    for category in df[column_name].unique():
        category_column = column_name + '_' + str(category)
        if category_column in df.columns:
            df = df.drop(category_column, axis=1)
        if category_column not in numerical_columns:
            numerical_columns.append(category_column)
    df= pd.concat([df,pd.get_dummies(df[column_name], prefix=column_name)],axis=1)
    return df
print(df.hist())
# Sex
sns.set(style="whitegrid")

g = sns.factorplot(x="Sex", y="Survived", data=df, size=4, palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
def get_sex_adult(row):
    age, sex = row
    if age < 18:
        return 'child'
    elif sex == 'female':
        return 'female_adult'
    else:
        return 'male_adult'

df['SexAdult'] = df[['Age', 'Sex']].apply(get_sex_adult, axis=1)
g = sns.factorplot(x="SexAdult", y="Survived", data=df, size=4, palette="muted")

if 'SexAdult' not in categorical_columns:
    categorical_columns.append('SexAdult')
# Embarked
df['Embarked'] = df['Embarked'].fillna('unknown')
if 'Embarked' not in categorical_columns:
    categorical_columns.append('Embarked')

df["Embarked_Category"] = pd.Categorical.from_array(df.Embarked).codes
if 'Embarked_Category' not in categorical_columns:
    categorical_columns.append('Embarked_Category')

g = sns.factorplot(x="Embarked_Category", y="Survived", data=df, size=4, palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
df_ticket = pd.DataFrame(df['Ticket'].value_counts())
df_ticket.rename(columns={'Ticket':'TicketMembers'}, inplace=True)

df_ticket['Ticket_perishing_women'] = df.Ticket[(df.SexAdult == 'female_adult')
                                               & (df.Survived == 0.0)
                                               & ((df.Parch > 0) | (df.SibSp > 0))].value_counts()
df_ticket['Ticket_perishing_women'] = df_ticket['Ticket_perishing_women'].fillna(0)
df_ticket['TicketGroup_include_perishing_women'] = df_ticket['Ticket_perishing_women'] > 0
df_ticket['TicketGroup_include_perishing_women'] = df_ticket['TicketGroup_include_perishing_women'].astype(int)

df_ticket['Ticket_surviving_men'] = df.Ticket[(df.SexAdult == 'male_adult')
                                              & (df.Survived == 1.0)
                                              & ((df.Parch > 0) | (df.SibSp > 0))].value_counts()
df_ticket['Ticket_surviving_men'] = df_ticket['Ticket_surviving_men'].fillna(0)
df_ticket['TicketGroup_include_surviving_men'] = df_ticket['Ticket_surviving_men'] > 0
df_ticket['TicketGroup_include_surviving_men'] = df_ticket['TicketGroup_include_surviving_men'].astype(int)

df_ticket["TicketId"] = pd.Categorical.from_array(df_ticket.index).codes
df_ticket.loc[df_ticket[df_ticket['TicketMembers'] < 3].index, "TicketId"] = -1
df_ticket["TicketMembers_Simple"] = pd.cut(df_ticket['TicketMembers'], bins=[0,1,4,20], labels=[0,1,2])
if 'TicketGroup_include_perishing_women' not in df.columns:
    df = pd.merge(df, df_ticket, left_on="Ticket", right_index=True, how='left', sort=False)

if 'Ticket_perishing_women' not in numerical_columns:
    numerical_columns.append('Ticket_perishing_women')
if 'TicketGroup_include_perishing_women' not in numerical_columns:
    numerical_columns.append('TicketGroup_include_perishing_women')
if 'Ticket_surviving_men' not in numerical_columns:
    numerical_columns.append('Ticket_surviving_men')
if 'TicketGroup_include_surviving_men' not in numerical_columns:
    numerical_columns.append('TicketGroup_include_surviving_men')
if 'TicketId' not in numerical_columns:
    numerical_columns.append('TicketId')
if 'TicketMembers' not in numerical_columns:
    numerical_columns.append('TicketMembers')
    
g = sns.factorplot(x="TicketGroup_include_perishing_women", y="Survived", data=df, size=4, palette="muted")
g = sns.factorplot(x="Ticket_surviving_men", y="Survived", data=df, size=4, palette="muted")
# surname
df['surname'] = df['Name'].apply(lambda x: x.split(',')[0].lower())
df_surname = pd.DataFrame(df['surname'].value_counts())
df_surname.rename(columns={'surname':'SurnameMembers'}, inplace=True)

df_surname['Surname_perishing_women'] = df.surname[(df.SexAdult == 'female_adult')
                                               & (df.Survived == 0.0)
                                               & ((df.Parch > 0) | (df.SibSp > 0))].value_counts()
df_surname['Surname_perishing_women'] = df_surname['Surname_perishing_women'].fillna(0)
df_surname['SurnameGroup_include_perishing_women'] = df_surname['Surname_perishing_women'] > 0
df_surname['SurnameGroup_include_perishing_women'] = df_surname['SurnameGroup_include_perishing_women'].astype(int)

df_surname['Surname_surviving_men'] = df.surname[(df.SexAdult == 'male_adult')
                                              & (df.Survived == 1.0)
                                              & ((df.Parch > 0) | (df.SibSp > 0))].value_counts()
df_surname['Surname_surviving_men'] = df_surname['Surname_surviving_men'].fillna(0)
df_surname['SurnameGroup_include_surviving_men'] = df_surname['Surname_surviving_men'] > 0
df_surname['SurnameGroup_include_surviving_men'] = df_surname['SurnameGroup_include_surviving_men'].astype(int)

df_surname["SurnameId"] = pd.Categorical.from_array(df_surname.index).codes
df_surname.loc[df_surname[df_surname['SurnameMembers'] < 3].index, "SurnameId"] = -1
df_surname["SurnameMembers_Simple"] = pd.cut(df_surname['SurnameMembers'], bins=[0,1,4,20], labels=[0,1,2])
if 'SurnameGroup_include_perishing_women' not in df.columns:
    df = pd.merge(df, df_surname, left_on="surname", right_index=True, how='left', sort=False)


if 'Surname_perishing_women' not in numerical_columns:
    numerical_columns.append('Surname_perishing_women')
if 'SurnameGroup_include_perishing_women' not in numerical_columns:
    numerical_columns.append('SurnameGroup_include_perishing_women')
if 'Surname_surviving_men' not in numerical_columns:
    numerical_columns.append('Surname_surviving_men')
if 'SurnameGroup_include_surviving_men' not in numerical_columns:
    numerical_columns.append('SurnameGroup_include_surviving_men')
if 'SurnameId' not in numerical_columns:
    numerical_columns.append('SurnameId')
if 'SurnameMembers' not in numerical_columns:
    numerical_columns.append('SurnameMembers')
    
g = sns.factorplot(x="SurnameGroup_include_perishing_women", y="Survived", data=df, size=4, palette="muted")
g = sns.factorplot(x="SurnameGroup_include_surviving_men", y="Survived", data=df, size=4, palette="muted")
# title
import re
df['Name_title'] = df['Name'].apply(lambda x: re.search(' ([A-Za-z]+)\.', x).group(1))
df.loc[df[df['Name_title'] == 'Ms'].index, 'Name_title'] = 'Miss'
print(df['Name_title'].unique())
if 'Name_title' not in categorical_columns:
    categorical_columns.append('Name_title')
g = sns.factorplot(y="Name_title", x="Survived", data=df, size=4, palette="muted")
title_mapping = {
    "Mr": 1, 
    "Miss": 2, 
    "Ms": 2, 
    "Mlle": 2, 
    "Mrs": 3, 
    "Mme": 3,
    "Master": 4, 
    "Dr": 5, 
    "Rev": 6, 
    "Major": 7, 
    "Capt": 7,
    "Col": 7, 
    "Don": 9,
    "Dona": 9, 
    "Sir": 9, 
    "Lady": 10, 
    "Countess": 10, 
    "Jonkheer": 10, 
}
df["Name_titleCategory"] = df.loc[:,'Name_title'].map(title_mapping)

if 'Name_titleCategory' not in categorical_columns:
    categorical_columns.append('Name_titleCategory')
g = sns.factorplot(x="Name_titleCategory", y="Survived", data=df, size=4, palette="muted")
# FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch']
if 'FamilySize' not in numerical_columns:
    numerical_columns.append('FamilySize')
g = sns.factorplot(x="SibSp", y="Survived", data=df, size=4, palette="muted")
g = sns.factorplot(x="Parch", y="Survived", data=df, size=4, palette="muted")
g = sns.factorplot(x="FamilySize", y="Survived", data=df, size=4, palette="muted")
# Name Length?
df['NameLength'] = df["Name"].apply(lambda x: len(x))
if 'NameLength' not in numerical_columns:
    numerical_columns.append('NameLength')
g = sns.factorplot(y="NameLength", x="Survived", data=df, size=4, palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
# Pclass
g = sns.factorplot(x="Pclass", y="Survived", data=df, size=4, palette="muted")
# cabin
# https://www.kaggle.com/c/titanic/prospector#1326
def get_cabin_location(cabin):
    if cabin == ' ':
        return 'no_cabin'
    # The cabin info consists of a letter (corresponding to a deck) 
    # and a cabin number, which is odd for cabins on the starboard side and even for the port.
    cabin_search_result = re.search('\d+', cabin)
    if cabin_search_result:
        type_code = np.int64(cabin_search_result.group(0))
        if type_code % 2 == 0:
            return 'port'
        else:
            return 'starboard'
    return 'unknown'

def get_cabin_deck(cabin):
    if cabin == ' ':
        return 'no_cabin'
    # The cabin info consists of a letter (corresponding to a deck) 
    # and a cabin number, which is odd for cabins on the starboard side and even for the port.
    cabin_search_result = re.search('[A-z]+', cabin)
    if cabin_search_result:
        return cabin_search_result.group(0)
    return 'unknown'

def get_cabin_count(cabin):
    if cabin == ' ':
        return 0
    cabin_search_result = re.findall('([A-z]\d+)', cabin)
    if cabin_search_result:
        return len(cabin_search_result)
    return 0

df['CabinLocation'] = df['Cabin'].fillna(' ').apply(get_cabin_location)
df['CabinDeck'] = df['Cabin'].fillna(' ').apply(get_cabin_deck)
df['CabinCount'] = df['Cabin'].fillna(' ').apply(get_cabin_count)

if 'CabinLocation' not in categorical_columns:
    categorical_columns.append('CabinLocation')
if 'CabinDeck' not in categorical_columns:
    categorical_columns.append('CabinDeck')
if 'CabinCount' not in numerical_columns:
    numerical_columns.append('CabinCount')

g = sns.factorplot(x="Survived", y="CabinLocation", data=df, size=4, palette="muted")
g = sns.factorplot(x="Survived", y="CabinDeck", data=df, size=4, palette="muted")
g = sns.factorplot(x="CabinCount", y="Survived", data=df, size=4, palette="muted")
df['CabinCategory'] = pd.Categorical.from_array(df.Cabin.fillna('0').apply(lambda x:x[0])).codes
g = sns.factorplot(y="Survived", x="CabinCategory", data=df, size=4, palette="muted")
if 'CabinCategory' not in categorical_columns:
    categorical_columns.append('CabinCategory')
# Fare
# df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df["Fare"] = df["Fare"].fillna(8.05)
print(df['Fare'].describe())
print(df['Fare'].hist())
g = sns.factorplot(x="Survived", y="Fare", data=df, size=4, palette="muted")
df['TicketMembers'] = df['TicketMembers'].fillna(0)
print(df.head()[['Pclass','Fare', 'TicketMembers']])
df['Fare_per_ticket_member'] = df['Fare'] / (df['TicketMembers'])
print(df['Fare_per_ticket_member'].hist())
g = sns.factorplot(x="Survived", y="Fare_per_ticket_member", data=df, size=4, palette="muted")
from math import log

class_fare = pd.DataFrame(columns=['count','mean','std','min','25%','50%','75%','max'])
class_fare.loc[1,:] = df[df['Pclass'] == 1]['Fare'].describe()
class_fare.loc[2,:] = df[df['Pclass'] == 2]['Fare'].describe()
class_fare.loc[3,:] = df[df['Pclass'] == 3]['Fare'].describe()

very_small_val = 0.01
df['Fare_standard_score_with_Pclass'] = df.apply(lambda row: (log(row['Fare'] + very_small_val) - log(class_fare.loc[row['Pclass'], 'mean'] + very_small_val)) / log(class_fare.loc[row['Pclass'], 'std'] + very_small_val), axis=1)
if 'Fare_standard_score_with_Pclass' not in numerical_columns:
    numerical_columns.append('Fare_standard_score_with_Pclass')
df[(df['Fare_standard_score_with_Pclass'] >= -0.5) & (df['Fare_standard_score_with_Pclass'] <= 0.5)]['Fare_standard_score_with_Pclass'].hist()
g = sns.factorplot(x="Survived", y="Fare_standard_score_with_Pclass", data=df, size=4, palette="muted")
from math import log

class_fare = pd.DataFrame(columns=['count','mean','std','min','25%','50%','75%','max'])
class_fare.loc[1,:] = df[df['Pclass'] == 1]['Fare_per_ticket_member'].describe()
class_fare.loc[2,:] = df[df['Pclass'] == 2]['Fare_per_ticket_member'].describe()
class_fare.loc[3,:] = df[df['Pclass'] == 3]['Fare_per_ticket_member'].describe()

very_small_val = 0.01
df['Fare_per_ticket_member_standard_score_with_Pclass'] = df.apply(lambda row: (log(row['Fare_per_ticket_member'] + very_small_val) - log(class_fare.loc[row['Pclass'], 'mean'] + very_small_val)) / log(class_fare.loc[row['Pclass'], 'std'] + very_small_val), axis=1)
if 'Fare_per_ticket_member_standard_score_with_Pclass' not in numerical_columns:
    numerical_columns.append('Fare_per_ticket_member_standard_score_with_Pclass')
df[(df['Fare_per_ticket_member_standard_score_with_Pclass'] >= -0.5) & (df['Fare_per_ticket_member_standard_score_with_Pclass'] <= 0.5)]['Fare_per_ticket_member_standard_score_with_Pclass'].hist()
g = sns.factorplot(x="Survived", y="Fare_per_ticket_member_standard_score_with_Pclass", data=df, size=4, palette="muted")
# https://www.kaggle.com/c/titanic/forums/t/11127/do-ticket-numbers-mean-anything
#print(df["Ticket"])
#print(df["Ticket"].value_counts())

def get_ticket_prefix(cabin):
    # The cabin info consists of a letter (corresponding to a deck) 
    # and a cabin number, which is odd for cabins on the starboard side and even for the port.
    cabin_search_result = re.search('[^\d]+', cabin)
    if cabin_search_result:
        return cabin_search_result.group(0).replace('/', '').replace('.', '').strip()
    return 'unknown'

df['TicketPrefix'] = df['Ticket'].apply(get_ticket_prefix)
g = sns.factorplot(y="TicketPrefix", x="Survived", data=df, size=8, palette="muted")

if 'TicketPrefix' not in categorical_columns:
    categorical_columns.append('TicketPrefix')
for col in categorical_columns:
    df = category_to_numeric(df, col)
# age prediction
from sklearn.ensemble import ExtraTreesRegressor

age_prediction_features = ['Fare', 'Fare_standard_score_with_Pclass',
                           #'Fare_per_ticket_member', 'Fare_per_ticket_member_standard_score_with_Pclass',
                           'Parch', 'Pclass', 'SibSp', 'Sex_female', 'Sex_male', 'FamilySize',
                           'NameLength', 'TicketMembers', 'TicketId', 
                           'Embarked_S', 'Embarked_C', 'Embarked_Q', 'Embarked_unknown', 
                           'Name_title_Mr', 'Name_title_Mrs', 'Name_title_Miss', 'Name_title_Master', 
                           'Name_title_Don', 'Name_title_Rev', 'Name_title_Dr', 'Name_title_Mme', 
                           'Name_title_Major', 'Name_title_Lady', 'Name_title_Sir', 'Name_title_Mlle', 'Name_title_Col', 
                           'Name_title_Capt', 'Name_title_Countess', 'Name_title_Jonkheer', 
                           'CabinLocation_no_cabin', 'CabinLocation_starboard', 'CabinLocation_port', 'CabinDeck_no_cabin', 
                           'CabinDeck_C', 'CabinDeck_E', 'CabinDeck_G', 'CabinDeck_D', 'CabinDeck_A', 'CabinDeck_B', 'CabinDeck_F', 'CabinDeck_T'
                          ]
age_prediction_tree_regressor = ExtraTreesRegressor(n_estimators=200)
age_X_train = df[age_prediction_features][df['Age'].notnull()]
age_Y_train = df['Age'][df['Age'].notnull()]
age_prediction_tree_regressor.fit(age_X_train, np.ravel(age_Y_train))

# predict only isnull values
df['Age_pred'] = df['Age']
df.loc[df[df['Age'].isnull()].index, 'Age_pred'] = age_prediction_tree_regressor.predict(df[age_prediction_features][df['Age'].isnull()])

if 'Age_pred' not in numerical_columns:
    numerical_columns.append('Age_pred')

# add ageGroup
df["AgeGroup"] = pd.cut(df['Age'], bins=[-2000,0,11,15,18,30,49,59,200], labels=[-1, 11,15,18,30,49,59,200])
df["AgeGroup_pred"] = pd.cut(df['Age_pred'], bins=[-2000,11,15,18,30,49,59,200], labels=[11,15,18,30,49,59,200])

if 'AgeGroup' not in numerical_columns:
    numerical_columns.append('AgeGroup')
if 'AgeGroup_pred' not in numerical_columns:
    numerical_columns.append('AgeGroup_pred')
    
g = sns.factorplot(y="Survived", x="AgeGroup", data=df, size=4, palette="muted")
g = sns.factorplot(y="Survived", x="AgeGroup_pred", data=df, size=4, palette="muted")
# Frugal_First_Class_Single_Man
# midle age first class single man with large discounted and unknown prefixed ticket and without cabin.
print("died", df[(df['Survived'] == 0) & (df['Sex'] == 'male') 
         & (df['Pclass'] == 1) 
         & (df['Age_pred'] <= 45) 
         & (df['Fare'] > 0)
         & (df['Fare_standard_score_with_Pclass'] < -0.25)
         & (df['TicketPrefix_unknown'] == 1)
         & (df['TicketMembers_Simple'] == 0)
         & (df['CabinCount'] == 0)
        ])
print("survived", df[(df['Survived'] == 1) & (df['Sex'] == 'male') 
         & (df['Pclass'] == 1) 
         & (df['Age_pred'] <= 45) 
         & (df['Fare'] > 0)
         & (df['Fare_standard_score_with_Pclass'] < -0.25)
         & (df['TicketPrefix_unknown'] == 1)
         & (df['TicketMembers_Simple'] == 0)
         & (df['CabinCount'] == 0)
        ])
df['Frugal_First_Class_Single_Man'] = 0

df.loc[df[(df['Sex'] == 'male') 
         & (df['CabinCount'] > 0)
         & (df['Embarked'] == 'C')
         & (df['SurnameMembers'] == 1)
         & (df['TicketPrefix_unknown'] == 1.0)
         & (df['Fare_standard_score_with_Pclass'] < -0.23)
         & (df['Pclass'] == 1)]['Frugal_First_Class_Single_Man'].index, 'Frugal_First_Class_Single_Man'] = 1
display(df[(df['Frugal_First_Class_Single_Man'] == 1)])
if 'Frugal_First_Class_Single_Man' not in numerical_columns:
    numerical_columns.append('Frugal_First_Class_Single_Man')
display(df[(df['Sex'] == 'female') & 
   (df['Fare_standard_score_with_Pclass'] <= -0.18) & 
   (df['Age_pred'] > 30) & 
   (df['Pclass'] == 3) & 
   (df['Name_title_Miss'] == 1.0)
  ])

# poor old miss
df['Poor_Old_Miss_Third_Class'] = 0
df.loc[df[(df['Sex'] == 'female') & 
   (df['Fare_standard_score_with_Pclass'] <= -0.18) & 
   (df['Age'] > 30) & 
   (df['Pclass'] == 3) & 
   (df['Name_title_Miss'] == 1.0)].index, 'Poor_Old_Miss_Third_Class'] = 1
       
if 'Poor_Old_Miss_Third_Class' not in numerical_columns:
    numerical_columns.append('Poor_Old_Miss_Third_Class')
display(df[(df['Sex'] == 'female') & 
   (df['Fare_standard_score_with_Pclass'] <= -0.18) & 
   (df['Age_pred'] >= 38) & 
   (df['Pclass'] == 2) & 
   (df['Name_title_Miss'] == 1.0) &
   (df['TicketPrefix_unknown'] == 1.0) &
   (df['SurnameMembers_Simple'] == 0)
  ])

# poor old miss
df['Poor_Old_Miss_Second_Class'] = 0
df.loc[df[
        (df['Sex'] == 'female') & 
        (df['Fare_standard_score_with_Pclass'] <= -0.18) & 
        (df['Age_pred'] >= 38) & 
        (df['Pclass'] == 2) & 
        (df['Name_title_Miss'] == 1.0) &
        (df['TicketPrefix_unknown'] == 1.0) &
        (df['SurnameMembers_Simple'] == 0)
         ].index, 'Poor_Old_Miss_Second_Class'] = 1
       
if 'Poor_Old_Miss_Second_Class' not in numerical_columns:
    numerical_columns.append('Poor_Old_Miss_Second_Class')
display(df[
    (df['Sex'] == 'female') & 
    (df['Fare_standard_score_with_Pclass'] <= -0.18) & 
    (df['Age_pred'] >= 35) & 
    (df['Pclass'] == 1) & 
    (df['Name_title_Miss'] == 1.0) &
    (df['SurnameMembers_Simple'] == 0)
  ])

# poor old miss
df['Poor_Old_Miss_First_Class'] = 0
df.loc[df[
            (df['Sex'] == 'female') & 
            (df['Fare_standard_score_with_Pclass'] <= -0.18) & 
            (df['Age_pred'] >= 35) & 
            (df['Pclass'] == 1) & 
            (df['Name_title_Miss'] == 1.0) &
            (df['SurnameMembers_Simple'] == 0)
         ].index, 'Poor_Old_Miss_First_Class'] = 1
       
if 'Poor_Old_Miss_First_Class' not in numerical_columns:
    numerical_columns.append('Poor_Old_Miss_First_Class')
df[(df['Sex'] == 'female') & (df['Fare'] <= 10) & (df['Age'] > 28) & (df['Name_title_Miss'] == 1.0)]

# poor old miss
df['Poor_Old_Miss'] = 0
df.loc[df[(df['Sex'] == 'female') 
         & (df['Fare'] <= 10) 
         & (df['Age_pred'] > 28) 
         & (df['Name_title_Miss'] == 1.0)].index, 'Poor_Old_Miss'] = 1
       
if 'Poor_Old_Miss' not in numerical_columns:
    numerical_columns.append('Poor_Old_Miss')
df[(df['Sex'] == 'female') & (df['Fare'] <= 10) & (df['Age'] > 26) & (df['Embarked'] == 'S') & (df['Name_title_Miss'] == 1.0)]

# poor Shouthampton old miss
df['Poor_Shouthampton_Old_Miss'] = 0
df.loc[df[(df['Sex'] == 'female') 
         & (df['Fare'] <= 10) 
         & (df['Age_pred'] > 26) 
         & (df['Embarked'] == 'S') 
         & (df['Name_title_Miss'] == 1.0)].index, 'Poor_Shouthampton_Old_Miss'] = 1
       
if 'Poor_Shouthampton_Old_Miss' not in numerical_columns:
    numerical_columns.append('Poor_Shouthampton_Old_Miss')
# feature selection
from sklearn.feature_selection import SelectKBest, f_classif

df_copied = df.copy()
df_copied['Name_titleCategory'] = df_copied['Name_titleCategory'].fillna(' ')
df_copied['Cabin'] = df_copied['Cabin'].fillna(' ')
df_copied['Age'] = df_copied['Age'].fillna(-300)
df_copied['AgeGroup'] = df_copied['AgeGroup'].fillna(-1.0)

train = df_copied[0:891].copy()
target = train["Survived"].values

selector = SelectKBest(f_classif, k=len(numerical_columns))
selector.fit(train[numerical_columns], target)
scores = -np.log10(selector.pvalues_)
indices = np.argsort(scores)[::-1]
print("Features importance :")
for f in range(len(scores)):
    print("%0.2f %s" % (scores[indices[f]],numerical_columns[indices[f]]))
# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

random_forest = RandomForestClassifier(n_estimators=3000, min_samples_split=4, class_weight={0:0.745, 1:0.255})
kfold = cross_validation.KFold(train.shape[0], n_folds=3, random_state=42)

scores = cross_validation.cross_val_score(random_forest, train[numerical_columns], target, cv=kfold)
print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean() * 100, scores.std() * 100, 'Random Forest Cross Validation'))

random_forest.fit(train[numerical_columns], target)
score = random_forest.score(train[numerical_columns], target)
print("Accuracy: %0.3f             [%s]" % (score * 100, 'Random Forest full test'))

importances = random_forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(len(numerical_columns)):
    print("%d. feature %d (%f) %s" % (f + 1, indices[f] + 1, importances[indices[f]] * 100, numerical_columns[indices[f]]))


features = [
            'Sex_female','Sex_male',
    
            'Age_pred',
    
            'SexAdult_male_adult','SexAdult_female_adult', 'SexAdult_child',
    
            'Name_titleCategory',
#             'Name_titleCategory_1',
#             'Name_titleCategory_2',
#             'Name_titleCategory_3',
#             'Name_titleCategory_4',
#             'Name_titleCategory_5',
#             'Name_titleCategory_6',
#             'Name_titleCategory_7',
#             'Name_titleCategory_9',
#             'Name_titleCategory_10',
#             'Name_title_Mr', 'Name_title_Mrs', 'Name_title_Miss', 'Name_title_Master', 
#             'Name_title_Don', 'Name_title_Rev', 'Name_title_Dr', 'Name_title_Mme', 
#             'Name_title_Major', 'Name_title_Lady', 'Name_title_Sir', 'Name_title_Mlle', 'Name_title_Col', 
#             'Name_title_Capt', 'Name_title_Countess', 'Name_title_Jonkheer', 

            'Pclass', 
            
            'TicketId',
    
            'NameLength',

            'CabinLocation_no_cabin', 'CabinLocation_starboard', 'CabinLocation_port', 
            'CabinCategory',
#             'CabinCategory_0',
#             'CabinCategory_1',
#             'CabinCategory_2',
#             'CabinCategory_3',
#             'CabinCategory_4',
#             'CabinCategory_5',
#             'CabinCategory_6',
#             'CabinCategory_7',
#             'CabinCategory_8',
#             'CabinDeck_C', 'CabinDeck_E', 'CabinDeck_G', 'CabinDeck_D', 'CabinDeck_A', 'CabinDeck_B', 'CabinDeck_F', 'CabinDeck_T','CabinDeck_no_cabin', 

            'SibSp','Parch',
    
            'Fare',
#             'Fare_per_ticket_member',
#             'Fare_standard_score_with_Pclass',
#             'Fare_per_ticket_member_standard_score_with_Pclass',
    
            'Embarked_Category',
#             'Embarked_S','Embarked_Q','Embarked_C','Embarked_unknown',
    
            'SurnameMembers_Simple','SurnameGroup_include_perishing_women','SurnameGroup_include_surviving_men',
    
            'TicketMembers_Simple', 'TicketGroup_include_perishing_women','TicketGroup_include_surviving_men',
    
            'FamilySize', 

#             'Frugal_First_Class_Single_Man',
#             'Poor_Old_Miss',
#             'Poor_Shouthampton_Old_Miss',
#             'Poor_Old_Miss_Third_Class',
#             'Poor_Old_Miss_Second_Class',
#             'Poor_Old_Miss_First_Class',
    
#             'TicketPrefix_SOPP', 'TicketPrefix_WC',
#             'TicketPrefix_unknown', 
#             'TicketPrefix_SCA','TicketPrefix_SP','TicketPrefix_SOP','TicketPrefix_Fa','TicketPrefix_SCOW','TicketPrefix_AS',
#             'TicketPrefix_FC','TicketPrefix_SOTONO','TicketPrefix_CASOTON','TicketPrefix_SWPP','TicketPrefix_SC','TicketPrefix_SCAH Basle',
    
#             'CabinCount',
           ]

# analyze failed.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, target, test_size=0.2, random_state=42)
random_forest = RandomForestClassifier(n_estimators=3000, min_samples_split=4, class_weight={0:0.745, 1:0.255})
kfold = cross_validation.KFold(X_train.shape[0], n_folds=3, random_state=42)

scores = cross_validation.cross_val_score(random_forest, X_train[features], y_train, cv=kfold)
print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean() * 100, scores.std() * 100, 'Random Forest Cross Validation'))

random_forest.fit(X_train[features], y_train)
score = random_forest.score(X_test[features], y_test)
print("Accuracy: %0.3f             [%s]" % (score * 100, 'Random Forest full test'))
pred_test = random_forest.predict(X_test[features])

importances = random_forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(len(features)):
    print("%d. feature %d (%f) %s" % (f + 1, indices[f] + 1, importances[indices[f]] * 100, features[indices[f]]))
pd.set_option("display.max_columns",101)
X_test_reseted = X_test.reset_index()
X_test_reseted['Survived_'] = y_test
X_test_reseted['Prediction'] = pred_test
X_test_reseted['pred_result'] = pred_test == y_test
display(X_test_reseted[(X_test_reseted['Survived'] == 1.0) & (X_test_reseted['pred_result'] == False)])
display(X_test_reseted[(X_test_reseted['Survived'] == 0.0) & (X_test_reseted['pred_result'] == False)])
# select specidic features
random_forest = RandomForestClassifier(n_estimators=3000, min_samples_split=4, class_weight={0:0.745, 1:0.255})
kfold = cross_validation.KFold(train.shape[0], n_folds=3, random_state=42)

scores = cross_validation.cross_val_score(random_forest, train[features], target, cv=kfold)
print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean() * 100, scores.std() * 100, 'Random Forest Cross Validation'))

random_forest.fit(train[features], target)
score = random_forest.score(train[features], target)
print("Accuracy: %0.3f             [%s]" % (score * 100, 'Random Forest full test'))

importances = random_forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(len(features)):
    print("%d. feature %d (%f) %s" % (f + 1, indices[f] + 1, importances[indices[f]] * 100, features[indices[f]]))
random_forest = RandomForestClassifier(n_estimators=3000, min_samples_split=4, class_weight={0:0.745, 1:0.255})
test = df_copied[891:].copy()
random_forest.fit(train[features], target)
predictions = random_forest.predict(test[features])
PassengerId = np.array(test["PassengerId"]).astype(int)
submit_df = pd.DataFrame(predictions, PassengerId, columns = ['Survived']).astype(int)
submit_df.to_csv('titanic.csv', index_label=['PassengerId'])
