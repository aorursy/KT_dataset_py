

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import random as random

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesClassifier

import sklearn.model_selection as model_selection

from sklearn.metrics import accuracy_score

import pandas_profiling

import warnings



warnings.filterwarnings('ignore')



seed = 12345

np.random.seed(seed)

random.seed(seed)



pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)



plt.style.use('ggplot')
#load the data

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



full = train.append(test)

full.sort_values('PassengerId', inplace=True)

full.reset_index(drop=True, inplace=True)
full.profile_report()
full[['Last','Full']] = full['Name'].str.split(r", ",expand=True)

full[['Title','Rest']] = full['Full'].str.split(r'(?<=\.)\s',n=1, expand=True)



full[['First', 'Parenthesis', 'blank']] = full['Rest'].str.split(r"\((.*)\)", expand=True)

full['blank'].unique()

full = full.drop('blank', axis=1)



full[['drop', 'Parenthesis2', 'blank']] = full['Rest'].str.split(r'\(\"(.*)\"\)', expand=True)

full['blank'].unique()

full = full.drop(['blank','drop'], axis=1)



full[['First', 'Quote', 'blank']] = full['First'].str.split(r"\"(.*)\"", expand=True)

full['Quote'].replace('', None, inplace=True)

full['blank'].unique()

full = full.drop('blank', axis=1)



#strip leading and trailing whitespaces

full['First'] = full['First'].str.strip()

full['Last'] = full['Last'].str.strip()

full['Title'] = full['Title'].str.strip()



full.drop(['Full', 'Rest'], axis=1, inplace=True)



full['hasNickname'] = ~full['Quote'].isnull()

full['hasParenthesis'] = ~full['Parenthesis'].isnull()

print(full.hasNickname.value_counts())

print(full.hasParenthesis.value_counts())

full = full.drop(['Parenthesis','Parenthesis2', 'Quote'], axis='columns')
cab_dum = full['Cabin'].str.get_dummies(' ')   # get dummies and split cabin by space at the same time

cab_occupancy =cab_dum.sum(axis=0)  # count cabin occupancy by taking row summation

occ_multiple = cab_occupancy[cab_occupancy.values>1].index

cab_cSum = cab_dum.sum(axis=1)  # n cabin per passenger

cab_multiple = cab_cSum[cab_cSum.values>1].index



full_cab = pd.concat([full, cab_dum.loc[:, occ_multiple]], axis=1)

##loop over each unique cabin to check whether there are different Ticket or Cabin values

df_temp = pd.DataFrame([])

for c in occ_multiple:



    n = full[full_cab[c]>0].Cabin.nunique()

    if n>1:

        df_temp = df_temp.append(full[full_cab[c]>0])



df_temp[['Cabin','Embarked', 'Fare', 'Pclass','Name', 'Ticket']].head(20)
full.Cabin = full.Cabin.str.replace('F ', 'F')
temp = full.groupby('Ticket').Cabin.nunique().sort_values(ascending=False)

print(temp.value_counts())
temp = full[full.groupby('Ticket').Cabin.transform(lambda x: x.fillna('').nunique())>1]

temp.sort_values(by=['Ticket','Cabin'], inplace=True)

temp[['Cabin','Embarked', 'Fare', 'Pclass','Name', 'Ticket']].head(10)
full['Cabin_Level'] = full['Cabin'].str[0]



df = full.groupby("Cabin_Level")['Pclass'].value_counts(normalize=False).unstack()

df.plot(kind='bar', stacked='True')

plt.title('Pclass proportion in each Cabin prefix alphabet')

plt.xticks(rotation=0)

plt.legend(title='Pclass', loc='center left', bbox_to_anchor=(1, 0.5))



# Fill in the missing value of Cabin and Cabin_Level

full['Cabin'] = full['Cabin'].fillna('')

full['Cabin_Level'] = full['Cabin_Level'].fillna('N')

full['Cabin_Level'] = full['Cabin_Level'].astype('category')



full.Cabin_Level.value_counts()
#full['ppFare'] = full.groupby('Ticket').Fare.transform(lambda x: x / x.count())

full['ppTicket'] = full.groupby('Ticket').Ticket.transform('count')  # how many person with the same Ticket number

full['ppFare'] = full.Fare / full.ppTicket # Fare price by person

#full['nLast'] = full.groupby('Last').Last.transform('count')



var = ['Fare', 'ppFare']

idvar = [e for e in list(full) if e not in var]

full_m = pd.melt(full, id_vars=idvar, value_vars=var)

g = sns.catplot(x = "variable", y='value', col="Pclass", row='Embarked', data=full_m, kind='strip', sharey=False)
full['Embarked'] = full.Embarked.fillna(full.Embarked.mode()[0])  # make a simple impute for Embarked

full['ppFare'] = full.groupby(['Pclass', 'Embarked']).ppFare.transform(lambda x: x.fillna(x.median()))

FareToFill = full.ppFare * full.ppTicket

full.loc[full.Fare.isna(), 'Fare'] = FareToFill[full.Fare.isna()]


x = full.Ticket.unique()

x.sort()

categories = [

  ('D7', "^[0-9]{7}$"),

  ('D6', "^[0-9]{6}$"),

  ('D5', "^[0-9]{5}$"),

  ('D4', "^[0-9]{4}$"),

  ('D3', "^[0-9]{3}$"),

  ('AxDn', "^A.*\s[0-9]{3,5}"),

  ('CxDn', "^C.*\s[0-9]{4,5}"),

  ('FxDn', "^F.*\s[0-9]{5}"),

  ('PxDn', "^P.*\s[0-9]{4,5}"),

  ('SOTON',"^(SOTON)|(STON)/O"),

  ('SxParis', "^S.*/P(ARIS|aris)"),

  ('SxOther', "^(?!.*(PARIS|Paris))((S\.)|(SC)|(SO\/)).*\s[0-9]{1,5}"),

  ('WxDn', "^W.*\s[0-9]{4,5}")

]



full['Ticket_Type'] = None

for c, matches in categories:

    full.loc[full.Ticket.str.match(matches), 'Ticket_Type'] = c



full['Ticket_Type'] = full.Ticket_Type.fillna('Other')



g = sns.catplot(x='Pclass', y='ppFare', col='Ticket_Type',col_wrap=4, hue='Embarked', kind='strip', data=full)
g = sns.FacetGrid(full, col="Pclass",row="Embarked",hue='Sex', margin_titles=True)

g.map(plt.scatter, 'Age','ppFare', alpha = 0.6).add_legend()
full['nFamily'] = full.groupby(['Last', 'Ticket_Type', 'Embarked']).Last.transform('count')



## Impute Age (using ExtraTreesRegressor)

to_impute = full[['Age', 'ppFare', 'Parch', 'Pclass', 'SibSp', 'hasNickname', 'hasParenthesis', 'ppTicket', 'nFamily']]

dummy = pd.get_dummies(data=full[['Title','Embarked', 'Sex','Ticket_Type']])

to_impute = pd.concat([to_impute,dummy], axis=1)

col = to_impute.columns



imp = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=100), max_iter=50, verbose=2)

imp.fit(to_impute)

imputed = pd.DataFrame(imp.transform(to_impute))

imputed.columns = col

imputed['PassengerId'] = full['PassengerId']

imputed['Age']=np.round(imputed['Age'], decimals=1)



full.update(imputed[['PassengerId','Age']])



full.info()
full['Title'] = full['Title'].replace(['Don.', 'Rev.', 'Jonkheer.', 'Sir.'], 'Honor')

full['Title'] = full['Title'].replace(['Dr.', 'Col.', 'Major.', 'Capt.'], 'Profession')

full['Title'] = full['Title'].replace(['Ms.', 'Mlle.', 'the Countess.', 'Lady.'], 'Miss.')

full['Title'] = full['Title'].replace(['Mme.', 'Dona.'], 'Mrs.')

full.Title.value_counts()



# Replace Cabin_Level 'T' with most frequent value from similar 'ppFare','Pclass', 'Embarked' group

full['Cabin_Level'] = full.groupby(['ppFare', 'Embarked', 'Pclass']).Cabin_Level.transform(lambda x: x.replace('T', x.mode()[0]))

full['Cabin_Level'].cat.remove_unused_categories(inplace=True)



print(full.Cabin_Level.value_counts())

#Merge Cabin_Level G(only 5 cases) with F

full['Cabin_Level'] = full['Cabin_Level'].replace('G', 'F')

full['Cabin_Level'].cat.remove_unused_categories(inplace=True)



# df = full.groupby("Cabin_Level")['Survived'].value_counts(normalize=True).unstack()

# df.plot(kind='bar', stacked='True')

# plt.xticks(rotation=0)

# plt.legend(title = 'Survived', bbox_to_anchor=(1, 0.5))
full['isMarried'] = full.Title.isin(['Mrs.', 'Mme.', 'Dona.'])

full['FamilySize'] = full ['SibSp'] + full['Parch'] + 1

full['CabinGroup'] =  full['Cabin_Level'].replace(['A','B'], 'AB')

full['CabinGroup'] =  full['CabinGroup'].replace(['C','F'], 'CF')

full['CabinGroup'] =  full['CabinGroup'].replace(['D','E'], 'DE')

full.CabinGroup.value_counts()



full.loc[(full['FamilySize']==1) & (full['ppTicket']==1) & (full['nFamily']==1), 'GroupCat'] = 'single'

full.loc[(full['FamilySize']==2) | (full['ppTicket']==2) | (full['nFamily']==2), 'GroupCat'] = 'couple'

full.loc[(full['FamilySize'].isin([3,4])) |(full['ppTicket'].isin([3,4])) | (full['nFamily'].isin([3,4])), 'GroupCat'] = 'small'

full.loc[(full['FamilySize']>4) | (full['ppTicket']>4) | (full['nFamily']>4), 'GroupCat'] = 'large'



full['Pclass_cat'] = full['Pclass'].astype('category')
mean_survival_rate = train.Survived.mean()



full['Family_code'] = full.groupby(['Last', 'ppFare']).ngroup()



train = full[~full['Survived'].isna()]

test = full[full['Survived'].isna()]



non_unique_families = [x for x in train['Family_code'].unique() if x in test['Family_code'].unique()]

non_unique_tickets = [x for x in train['Ticket'].unique() if x in test['Ticket'].unique()]

non_unique_cabins = [x for x in train['Cabin'].unique() if x in test['Cabin'].unique()]



non_unique_cabins.remove("")



full['Family_Survive'] = full.groupby('Family_code').Survived.transform('median')  ##try max maybe (if any one survived)

full['Ticket_Survive'] = full.groupby(['Ticket']).Survived.transform('median')     #try max maybe

full['Cabin_Survive'] = full.groupby(['Cabin']).Survived.transform('median')     #try max maybe



full['non_unique_families'] = full.Family_code.isin(non_unique_families)

full['non_unique_tickets'] = full.Ticket.isin(non_unique_tickets)

full['non_unique_cabins'] = full.Cabin.isin(non_unique_cabins)





full.loc[~full.non_unique_families,'Family_Survive'] = np.nan

full.loc[~full.non_unique_tickets,'Ticket_Survive'] = np.nan

full.loc[~full.non_unique_cabins,'Cabin_Survive'] = np.nan





full['Survival_Rate'] = full.Family_Survive.fillna(full.Ticket_Survive.fillna(full.Cabin_Survive.fillna(mean_survival_rate)))

full['hasSurvivalInfo'] = full.non_unique_tickets | full.non_unique_families | full.non_unique_cabins



full['Survival_Rate'] = full.Survival_Rate.fillna(mean_survival_rate)

full['Survival_Rate'] = np.round(full.Survival_Rate, decimals=2)

full['Survival_Rate_cat'] = full['Survival_Rate'].astype('category')
g = sns.catplot(data=full, y='Survived', x='Survival_Rate_cat', col='Sex', kind='violin')
cat = full.select_dtypes(include=['object','bool','category', 'int64']).columns

cat = cat.drop(['Cabin','Name','PassengerId', 'Ticket', 'Last', 'First', 'Family_code', 'non_unique_families', 'non_unique_tickets', 'non_unique_cabins'])



fig, axes = plt.subplots(4, 4, figsize=(15, 10))

for idx, (c, ax) in enumerate(zip(cat, axes.flatten())):

    sns.barplot(data = full, y='Survived', x=c, ax=ax)

    plt.tight_layout()

else:

    [ax.set_visible(False) for ax in axes.flatten()[idx+1:]]



#plot the numerical variables

num = full.select_dtypes(include=['float']).columns

num = num.drop(['Survived'])



fig, axes = plt.subplots(2, 4, figsize=(15, 10), sharey=False)

for idx, (c, ax) in enumerate(zip(num, axes.flatten())):

    sns.violinplot(data = full, x='Survived', y=c, ax=ax)

    plt.tight_layout()

else:

    [ax.set_visible(False) for ax in axes.flatten()[idx+1:]]
## now drop those temporary columns used for processing

full2 = full[['PassengerId','Survived', 'Age','Pclass_cat', 'Embarked', 'Sex', 'Title', 'Cabin_Level',  'ppTicket', 'ppFare', 'Survival_Rate', 'isMarried', 'hasSurvivalInfo', 'FamilySize', 'Parch', 'SibSp', 'nFamily', 'GroupCat']]

dummyVars = full2.select_dtypes(include=['object','bool','category']).columns

full3 = pd.get_dummies(full2, columns = dummyVars, drop_first=True)



train = full3[~full3['Survived'].isna()]

test = full3[full3['Survived'].isna()]

X_train = train.drop(["Survived","PassengerId"],axis=1)

Y_train = train['Survived'].astype(int)

X_test = test.drop(["Survived","PassengerId"],axis=1)
clf_ET = ExtraTreesClassifier(random_state=0, bootstrap=True, oob_score=True)



sss = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.33, random_state= 0)

sss.get_n_splits(X_train, Y_train)



parameters = {'n_estimators' : np.r_[10:210:10],

              'max_depth': np.r_[1:6]

             }



grid = model_selection.GridSearchCV(clf_ET, param_grid=parameters, scoring = 'accuracy', cv = sss, return_train_score=True, n_jobs=4, verbose=2)

grid.fit(X_train,Y_train)
a = pd.DataFrame(grid.cv_results_)

params = a.columns[a.columns.str.contains('param_')]

m = pd.melt(a, id_vars=params, value_vars='mean_test_score')



plt.figure(figsize=(5,5))

g = sns.lineplot(data=m, x=params[1], y='value', hue=params[0], palette='Set1')

g.legend( bbox_to_anchor=(1, 0.5))
p = a[a.param_max_depth==4].sort_values(by='mean_test_score', ascending=False).iloc[0]

print("Best mean test score: %f using %s" % (p.mean_test_score, p.params))



bst_ET = grid.best_estimator_

bst_ET.set_params(**p.params)

bst_ET.fit(X_train,Y_train)

pred_ET = bst_ET.predict(X_test)



test['Survived'] = pred_ET.astype(int)

submission = test[['PassengerId', 'Survived']]

submission.PassengerId = submission.PassengerId.astype(int)

submission.sort_values(by=['PassengerId'], inplace=True)

submission.to_csv('pred_final.csv', index=False)



submission.head(10)