#Imports

import numpy as np

import pandas as pd



#Plotting

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#Modeling

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



#Misc

from __future__ import division, print_function
#Read Input Data

X_train = pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv',index_col=0)

X_train.head()
X_train.info()
#Custom Describe!

def var_summary(x):

    return pd.Series([x.count(), x.isnull().sum(),x.mean(), x.std(),x.min(),x.quantile(0.01), x.quantile(0.25), 

                     x.quantile(0.5),x.quantile(0.75),x.quantile(.99),x.max()], 

                    index=['Count','Miss','Mean','Std','Min','1%','25%','50%','75%','99%','Max'])



X_train.select_dtypes(exclude=['object']).apply(lambda x: var_summary(x)).T
X_train.describe(include=['object'])
#Categorize Cabin Feature and Handle Missing Values

X_train['Cabin'] = X_train.Cabin.fillna('U').map(lambda x: x[0])

X_test['Cabin'] = X_test.Cabin.fillna('U').map(lambda x: x[0])
#Since there is a difference between test and train data in no. of distinct categories of Cabin 

# as per our Definition above, replacing T with U as count is only 1 in training data whereas 

#it is missing in Testing!



print (sorted(X_train.Cabin.unique()))

print (sorted(X_test.Cabin.unique()))



X_train.loc[X_train.Cabin =='T','Cabin'] = 'U'

print (X_train.Cabin.value_counts())
#Dropping Unwanted Columns as per analysis. (We could have analysed Ticket Feature also .. 

#but in given test sample, its all NaN! :P)

X_train.drop(['PassengerId','Ticket'],axis=1,inplace=True)

X_test.drop(['Ticket'], axis=1, inplace=True)

X_train.head()
#Missing Values Handling:



#Age 



#(Training)

avg_age = X_train.Age.mean()

std_age = X_train.Age.std()

missing_age_count = X_train.Age.isnull().sum()

random_age_list = np.random.randint(avg_age - std_age, avg_age + std_age, size = missing_age_count)



X_train.loc[X_train.Age.isnull(),'Age'] = random_age_list



#(Testing)

avg_age = X_test.Age.mean()

std_age = X_test.Age.std()

missing_age_count = X_test.Age.isnull().sum()

random_age_list = np.random.randint(avg_age - std_age, avg_age + std_age, size = missing_age_count)



X_test.loc[X_test.Age.isnull(),'Age'] = random_age_list







#Embarked (Not required for test data)

X_train['Embarked'].fillna('S',inplace=True)





#Fare  (Not required for training data)

X_test.Fare.fillna(X_test.Fare.mean(),inplace=True)
sns.distplot(X_train.Age,bins=50)
#Considering above plot, creating new feature for age below 16

X_train['Is_Child'] = (X_train.Age < 16).astype(int)

X_test['Is_Child'] = (X_test.Age < 16).astype(int)
#Name Feature

X_train['Name_Len'] = X_train.Name.str.len()

X_train.drop('Name',axis=1, inplace=True)



X_test['Name_Len'] = X_test.Name.str.len()

X_test.drop('Name',axis=1, inplace=True)
#As per percentile analysis of both the samples, handling outliers!



X_train['Fare'] = X_train.Fare.clip_lower(X_train.Fare.quantile(0.05))

X_train['Fare'] = X_train.Fare.clip_upper(X_train.Fare.quantile(0.99))



X_test['Fare'] = X_test.Fare.clip_lower(X_test.Fare.quantile(0.05))

X_test['Fare'] = X_test.Fare.clip_upper(X_test.Fare.quantile(0.99))



#Fare vs Survived Analysis

fare_survived = X_train.loc[X_train.Survived == 1,'Fare']

fare_not_survived = X_train.loc[X_train.Survived == 0,'Fare']



# get average and std for fare of survived/not survived passengers

avgerage_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare      = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])



avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
#Categorise Fare into dummy variables using quantile ranges



#Training

X_train['Fare_1'] = (X_train.Fare < 7.91).astype(int)

X_train['Fare_2'] = ((X_train.Fare > 7.91) & (X_train.Fare < 14.45)).astype(int)

X_train['Fare_3'] = ((X_train.Fare > 14.45) & (X_train.Fare < 31.50)).astype(int)

X_train['Fare_4'] = (X_train.Fare > 31.50).astype(int)



#Test

X_test['Fare_1'] = (X_test.Fare < 7.91).astype(int)

X_test['Fare_2'] = ((X_test.Fare > 7.91) & (X_test.Fare < 14.45)).astype(int)

X_test['Fare_3'] = ((X_test.Fare > 14.45) & (X_test.Fare < 31.50)).astype(int)

X_test['Fare_4'] = (X_test.Fare > 31.50).astype(int)



#Drop Fare now.. as it will be highly co-rrelated with above features

X_train.drop('Fare',axis=1, inplace=True)

X_test.drop('Fare',axis=1, inplace=True)
#Add Family Feature using Parch & SibSp & Binarize. 



#Training

X_train['Family'] = X_train.Parch + X_train.SibSp

X_train.loc[X_train.Family > 0 ,'Family'] = 1

X_train.loc[X_train.Family == 0, 'Family'] = 0



X_train.drop(['Parch','SibSp'],axis=1, inplace=True)





#Test

X_test['Family'] = X_test.Parch + X_test.SibSp

X_test.loc[X_test.Family > 0 ,'Family'] = 1

X_test.loc[X_test.Family == 0, 'Family'] = 0





X_test.drop(['Parch','SibSp'],axis=1, inplace=True)
sns.countplot(x=X_train.Family)

family_perc = X_train[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0])
#Creation of Dummy Variables/Features for remaining Categorical Features.



X_train['Sex'] = X_train.Sex.map({'male':1 , 'female':0})

X_test['Sex'] = X_test.Sex.map({'male':1 , 'female':0})



#Dropping one of the category from each feature using drop_first=True

X_train = pd.get_dummies(X_train, columns=['Pclass','Embarked','Cabin'],drop_first=True)

X_test = pd.get_dummies(X_test, columns=['Pclass','Embarked','Cabin'],drop_first=True)
#Plotting pearson correlation matrix using HeatMap. Sorting it based on correlation!



corrmat = X_train.corr()



k = 10 #number of variables for heatmap



plt.figure(figsize=(15,15))

cols = corrmat.nlargest(k, 'Survived')['Survived'].index

cm = np.corrcoef(X_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 

                 annot_kws={'size': 10},

                 cmap=plt.cm.viridis, 

                 yticklabels=cols.values, 

                 linecolor='white',

                 xticklabels=cols.values)

plt.show()
#Creating Dependent Variable Series for modelling and dropping it from X_train

Y_train = X_train.Survived

X_train.drop(['Survived'],axis=1,inplace=True)
#sklearn Feature Selection option



#skb = SelectKBest(k=10)

#selected_features = skb.fit(X_train, Y_train)

#indices_selected = selected_features.get_support(indices=True)

#columns_selected = [col for i, col in enumerate(X_train.columns) if i in indices_selected]

#columns_selected
#X_train = X_train[columns_selected]

#X_test = X_test[columns_selected]
#Logistic Regression

log_reg = LogisticRegression()

log_reg.fit(X_train,Y_train)

log_reg.score(X_train,Y_train)
#Random Forest Classifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, Y_train)

rfc.score(X_train,Y_train)
print ('Cross Validation Score Using : Logistic Regression')

print (cross_val_score(log_reg, X_train, Y_train,cv=10))

print ('#' * 80)

print (np.mean(cross_val_score(log_reg, X_train, Y_train,cv=10)))
print ('Cross Validation Score Using : Random Forest Classifier')

print (cross_val_score(rfc, X_train, Y_train,cv=10))

print ('#' * 80)

print (np.mean(cross_val_score(rfc, X_train, Y_train,cv=10)))
#Predict & Store it in CSV as per kaggle requested format!

Y_pred = rfc.predict(X_test)

pd.DataFrame({'PassengerId':X_test.index, 'Survived':Y_pred}).set_index('PassengerId').to_csv('submission1.csv')