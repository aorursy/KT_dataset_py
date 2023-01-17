# load packages

import numpy as np # data processing

import pandas as pd # data processing

import matplotlib.pyplot as plt # visualizaion

import seaborn as sns # visualization



# display graphs inline

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

full = pd.concat([train,test])

full.info()
# Grab title from passenger names

def substrTitle(name):

    start = name.find(', ')

    end = name.find('.')

    return name[start+2:end]



full["Title"] = full.Name.map(substrTitle)



# Show title counts by sex

pd.crosstab(full.Sex,full.Title)
# Titles with very low cell counts to be combined to "rare" level

rare_title = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']



# Also reassign mlle, ms, and mme accordingly

full.loc[:,"Title"] = full["Title"].map(lambda x: x if x not in rare_title else 'Rare_Title')

full.loc[full.Title == 'Mlle',"Title"] = 'Miss'

full.loc[full.Title == 'Ms',"Title"] = 'Miss'

full.loc[full.Title == 'Mme',"Title"] = 'Mrs'



# Show title counts by sex again

pd.crosstab(full.Sex,full.Title)
# Finally, grab surname from passenger name

def substrSurname(name):

    start = name.find('. ')

    return name[start+2:].strip()



full["Surname"] = full.Name.map(substrSurname)



print('We have', len(np.unique(full.Surname)), 'unique surnames. I would be interested to infer ethnicity based on surname --- another time.')
# Create a family size variable including the passenger themselves

full["Fsize"] = full.SibSp + full.Parch + 1



# Create a family variable 

full["Family"] = full.Surname + '_' + str(full.Fsize)
# Use Seaborn to isualize the relationship between family size & survival

sns.set_style('white')

fig,ax = plt.subplots(figsize=(8,6))

conntplot = sns.countplot(x="Fsize",hue='Survived',data=full)

legend = plt.xlabel('Family Size')
# Discretize family size

full["FsizeD"] = None

full.loc[full.Fsize == 1,"FsizeD"] = 'singleton'

full.loc[(full.Fsize < 5) & (full.Fsize > 1),"FsizeD"] = 'small'

full.loc[full.Fsize > 4,"FsizeD"] = 'large'



from statsmodels.graphics.mosaicplot import mosaic

fig,ax = plt.subplots(figsize=(8,6))

m = mosaic(full, ['FsizeD', 'Survived'],title='Family Size by Survival',ax=ax)
# This variable appears to have a lot of missing values

full.Cabin[0:28]
# The first character is the deck. For example:

full.Cabin[1:2]
# Create a Deck variable. Get passenger deck A - F:

def subStrDeck(cabin):

    if not cabin or pd.isnull(cabin):

        return None

    else:

        return cabin[0]



full["Deck"] = full.Cabin.map(subStrDeck)
# Passengers 62 and 830 are missing Embarkment

full['Embarked'].iloc[[61,829]]
# Get rid of our missing passenger IDs

embark_fare = full[(full.PassengerId!=61) & (full.PassengerId != 829)]

fig,ax = plt.subplots(figsize=(8,6))

b = sns.boxplot(x='Embarked',y='Fare',hue='Pclass',data=full)

line = ax.axhline(y=80, color='r', linestyle='--')
# Since their fare was $80 for 1st class, they most likely embarked from 'C'

full.loc[[61,829],'Embarked'] = 'C'
# Show row 1044

full.iloc[1043]
fig,ax = plt.subplots(figsize=(8,6))

distp = sns.distplot(full.loc[(full.Pclass == 3) & (full.Embarked == 'S'),'Fare'],hist=False)

line = ax.axvline(x=np.nanmedian(full.loc[(full.Pclass == 3) & (full.Embarked == 'S'),'Fare']),color='r', linestyle='--')

plt.ylabel('Density')
# Replace missing fare value with median fare for class/embarkment

full.iloc[1043,3] = 8.05
# Show number of missing Age values

sum(np.isnan(full.Age))
full["Deck"] = full.Deck.astype('category')

full["Deck"].cat.add_categories("Missing")

full["Deck"] = full.Deck.cat.codes
from sklearn.preprocessing import Imputer



imp = Imputer(missing_values='NaN', strategy='median', axis=0)

full["Age"] = imp.fit_transform(full.Age.reshape(-1,1))



imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

full["Deck"] = imp.fit_transform(full.Deck.reshape(-1,1))
# First we'll look at the relationship between age & survival

# I include Sex since we know (a priori) it's a significant predictor

plt.figure(figsize=(12,8))

plt.subplot(121)

x1 = full[(full.Sex=='male') & (full.Survived==1)].Age

x2 = full[(full.Sex=='male') & (full.Survived==0)].Age

plt.hist([x1,x2],stacked=True)

plt.title('female')

plt.subplot(122)

x1 = full[(full.Sex=='female') & (full.Survived==1)].Age

x2 = full[(full.Sex=='female') & (full.Survived==0)].Age

plt.hist([x1,x2], stacked=True)

plt.title('female')
# Create the column child, and indicate whether child or adult

full.loc[full.Age < 18,'Child'] = 'Child'

full.loc[full.Age >= 18,'Child'] = 'Adult'



# Show counts

pd.crosstab(full.Child,full.Survived)
# Adding Mother variable

full["Mother"] = 'Not Mother'

full.loc[(full.Sex == 'female') & (full.Parch > 0) & (full.Age > 18) & (full.Title != 'Miss'),'Mother'] = 'Mother'



# Show counts

pd.crosstab(full.Mother,full.Survived)
full.info()
# delete unnecessary variables

full.drop(['Name','Cabin','Ticket','PassengerId'],axis=1)



# Set the seed

quality_features = full.select_dtypes(include=['object'])

quantity_features = full.select_dtypes(exclude=['object'])

quality_encoded = pd.get_dummies(quality_features)



full_final = pd.concat([quantity_features,quality_encoded],axis=1)
train = full_final[0:891]

test = full_final[891:1309]
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)

clf.fit(train.loc[:,(train.columns != 'Survived') & (train.columns != 'PassengerId')],train['Survived'])
# extract importances of features

estimator_importances = clf.feature_importances_

# construct the Series data structure for plotting

estimators = train.loc[:,(train.columns != 'Survived') & (train.columns != 'PassengerId')].columns

importances = pd.Series(estimator_importances,index=estimators)

# sort the featue by importances

importances.sort_values(ascending=False,inplace=True)

importances[:10].plot(kind='barh')
# predict using the test set

output = clf.predict(test.loc[:,(train.columns != 'Survived') & (train.columns != 'PassengerId')])



# save the solution to file

test.loc[:,"Survived"] = list(output)

test[["PassengerId","Survived"]].to_csv('resu.csv',header=True,index=False)