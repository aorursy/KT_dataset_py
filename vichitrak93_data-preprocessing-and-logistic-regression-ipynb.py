%matplotlib inline
import pandas as pd
pd.options.display.max_columns = 150
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import sys
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
#General Analysis
train_df.head()

print(train_df.shape)
train_df.describe()
#Age has only 714 values out of 891. We need to fill the null values on Age column
for data_frame in [train_df, test_df]:
    data_frame['Age'] = data_frame['Age'].fillna(data_frame['Age'].median())

#make some plots
fig = plt.figure(figsize=(25,7))
sns.violinplot(x='Sex',y='Age',hue='Survived',data=train_df,split=True,palette={0:"r",1:"g"});
#Conclusion : females survived more than males
#             Adult females survived more than males
#combine train and test dataframes

def combined_dataframes():
    
    train_df = pd.read_csv('./input/train.csv')
    test_df = pd.read_csv('./input/test.csv')
    
    #test dataset doesn't contain the target variable column
    #extract and remove the target variable from train dataset
    target_var = train_df.Survived
    train_df.drop(['Survived'],1,inplace=True)
    
    #Merge train and test datasets
    combined_dataset = train_df.append(test_df)
    
    combined_dataset.reset_index(inplace=True)
    combined_dataset.drop(['index','PassengerId'], inplace=True, axis=1)
    
    return combined_dataset
combined_df = combined_dataframes()
combined_df.head()
#1309 is exact number of sum of train and test row dataset
titles_set=set()
for name in train_df['Name']:
    titles_set.add(name.split(',')[1].split('.')[0].strip())

print(titles_set)
Title_Dict = {"Capt": "Officer", "Col": "Officer", "Major": "Officer", "Jonkheer": "Royalty", "Don": "Royalty", "Sir" : "Royalty",
                    "Dr": "Officer", "Rev": "Officer", "the Countess":"Royalty", "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs", "Mr" : "Mr",
                    "Mrs" : "Mrs", "Miss" : "Miss", "Master" : "Master", "Lady" : "Royalty"}
def extract_titles():
    combined_df['titles'] = combined_df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    combined_df['titles'] = combined_df.titles.map(Title_Dict)
    return combined_df
combined_df = extract_titles()
combined_df.head(2)
combined_df[combined_df['titles'].isnull()]
#Process name values
def handle_namesValues():
    global combined_df
    #remove the Names columnas we have the titles and name won't do much for the classification
    combined_df.drop(['Name'],axis=1,inplace=True)
    #create dummy variables for titles columns
    title_dummies = pd.get_dummies(combined_df['titles'],prefix = 'titles')
    combined_df = pd.concat([combined_df, title_dummies],axis=1)
    combined_df.drop(['titles'],axis=1,inplace=True)
    return combined_df
combined_df = handle_namesValues()
combined_df.head()
#check if Fare column has any missing value
combined_df[combined_df['Fare'].isnull()]
def handle_fare():
    global combined_df
    combined_df.Fare.fillna(combined_df.iloc[:891].Fare.mean(),inplace = True)
    return combined_df
combined_df = handle_fare()
#Check if embark column has any missing values
combined_df[combined_df['Embarked'].isnull()]
def handle_embarked():
    global combined_df
    combined_df.Embarked.fillna('S',inplace = True)
    #dummy encoding for embarked values
    embarked_dummies = pd.get_dummies(combined_df['Embarked'],prefix = 'Embarked')
    combined_df = pd.concat([combined_df,embarked_dummies],axis = 1)
    combined_df.drop('Embarked',axis=1,inplace = True)
    return combined_df
combined_df = handle_embarked()
combined_df.head()
#combined_df[combined_df['Cabin'].isnull()]
def handle_sex():
    global combined_df
    combined_df['Sex'] = combined_df['Sex'].map({'male':1,'female':0})
    return combined_df
combined_df = handle_sex()
combined_df.head()
def handle_Pclass():
    global combined_df
    Pclass_dummies = pd.get_dummies(combined_df['Pclass'], prefix = "Pclass")
    #get dummy variables for Pclass column
    combined_df  = pd.concat([combined_df, Pclass_dummies], axis=1)
    combined_df.drop('Pclass', axis=1, inplace = True)
    return combined_df
combined_df = handle_Pclass()
combined_df.head()
def handle_family():
    global combined_df
    combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1
    return combined_df
combined_df = handle_family()
combined_df.head()
combined_df = combined_df.drop(['Ticket','Cabin'], axis=1)
combined_df.head()
def handle_Age():
    average_age = combined_df['Age'].mean()
    std_age = combined_df['Age'].std()
    nan_age_count = combined_df['Age'].isnull().sum()
    
    random_age_value = np.random.randint(average_age - std_age, average_age + std_age, size = nan_age_count)
    combined_df['Age'][np.isnan(combined_df['Age'])] = random_age_value
    combined_df['Age'] = combined_df['Age'].astype(int)
    return combined_df
combined_df = handle_Age()
combined_df[combined_df['Age'].isnull()]
def break_train_test_sets():
    global combined_df
    X_train = combined_df.iloc[:891]
    Y_train = pd.read_csv('./input/train.csv', usecols=['Survived'])['Survived'].values
    X_test = combined_df.iloc[891:]
    return X_train, Y_train, X_test
X_train, Y_train, X_test = break_train_test_sets()
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
logreg.score(X_train, Y_train)
submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": Y_pred})

#Put it in csv file
submission.to_csv('./input/submission.csv', index=False)
