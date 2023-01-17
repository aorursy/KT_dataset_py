# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from scipy import stats

from itertools import permutations, combinations



sns.set(color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
titanic = pd.read_csv("../input/train.csv")
titanic.info()
titanic.head()
fig, ax = plt.subplots(1,1,figsize=(8,8))

age = titanic.Age.dropna()

ax.hist(age);

ax.axvline(age.mean(), color='r',linestyle='--');

ax.set_title('Age Range on Titanic', size=14);
titanic.Age.describe()
fig, ax = plt.subplots(1,1,figsize=(8,8))

color=['b','g','r']

for bar,c in  zip(range(3),color):

    ax.bar(titanic.Pclass.value_counts().sort_index().index[bar],titanic.Pclass.value_counts().sort_index().values[bar], color=c);

plt.legend(['First Class','Second Class','Third Class']);
fig,ax = plt.subplots(1,1,figsize=(7,7))

titanic.Sex.value_counts()

sex =['Male','Female']

ax.pie(titanic.Sex.value_counts(),explode=(0,0.05),

       labels=sex,

       autopct='%1.1f%%', 

       startangle=90);

ax.legend(sex);

ax.set_title('Sex',size=15, loc='center');
fig,ax = plt.subplots(1,1,figsize=(7,7))

titanic.Sex.value_counts()

survived =['No','Yes']

ax.pie(titanic.Survived.value_counts(),explode=(0,0.05),

       labels=survived,

       colors=['r','b'],

       autopct='%1.1f%%', 

       startangle=90);

ax.legend(['No','Yes']);

ax.set_title('Survived',size=14);
fig,ax = plt.subplots(1,1,figsize=(8,5))

fares =titanic.Fare

ax.hist(fares);

ax.set_title('Fares',size=15);

print('The Mean fare was {} and the Standard Deviation is {}.'.format(round(fares.mean(),2),round(fares.std(),2)))

print('The most expensive ticket costed {} and the cheapest went for {}.'.format(fares.max(),fares.min()))

#Why is the cheapest fare free? (0)
fig,axes = plt.subplots(1,2,figsize=(14,7))

data = ['SibSp','Parch']

colors = ['b','g']

for ax,col,c in zip(axes,data,colors):

    ax.hist(titanic[col],color=c);

    ax.set_title(col);
sns.pairplot(titanic[['Fare','Survived','Pclass','SibSp']]);
#needed data for chi2

survival_by_class = titanic.groupby('Pclass').Survived.value_counts()

deaths_by_class = survival_by_class[:,0]

survived_by_class = survival_by_class[:,1]
fig, ax = plt.subplots(1,1,figsize=(8,8));

(survival_by_class.sort_index(ascending=True)).plot(kind='bar');

ax.set_xticklabels(['Upper Class Died','Upper Class Survived',

                   'Middle Class Died','Middle Class Survived',

                   'Lower Class Death','Lower Class Survived'],size=13);

ax.set_title('')

survival_by_class.sort_index(ascending=False);

plt.xticks(rotation=45);
print(survived_by_class)

print(deaths_by_class)

chi2, p, dof, expected = stats.chi2_contingency([survived_by_class,deaths_by_class], correction=False)
p #The probabilty that deaths were not related to class and by sheer coincid`ence is 0
def by_embark(col):

    return titanic[titanic.Embarked.notnull()].groupby('Embarked')[col].mean()
print('Locations: Cherbourg, Queenstown, Southampton ')

print((titanic[titanic.Embarked.notnull()].Embarked.value_counts()).sort_values().sort_index())

print('\nFare')

print(by_embark('Fare').sort_index())

print('\nSurvived')

print(by_embark('Survived').sort_index())

print('\nPclass')

print(by_embark('Pclass').sort_index())

print('\nSex')

print(titanic[titanic.Embarked.notnull()].groupby('Embarked').Sex.value_counts().sort_index())

print('\nSibSp')

print(by_embark('SibSp').sort_index())

print('\nParch')

print(by_embark('Parch').sort_index())

Fare_by_embark = titanic[titanic.Embarked.notnull()].groupby('Embarked').Fare.mean()
def numeric_col(df):

    """1st step, Takes a dataframe and returns a combination of 2 columns that are numeric"""

    numeric_columns = []

    for col in df.columns:

        df_dtype = df[col].dtypes

        if df_dtype in ['int', 'float']:

            numeric_columns.append(col)

    numeric_combos = combinations(numeric_columns,2)

    return numeric_combos
def mututal_data(col_combo,df):

    """Gets the index that contains no NaN for any two columns, col_combos is a tuple from numeric_col, ACCESSED BY col_valid_index"""

    col1,col2 = col_combo

    length = len(df)

    col1_valid = df[col1].notnull()==True

    col2_valid = df[col2].notnull()==True

    len1 = len(df[col1_valid])

    len2 = len(df[col2_valid])

    if length==len1 and length==len2:

        return df.index

    elif len1!=length and len2!=length:

        maskx= df[col1_valid].index

        masky= df[col2_valid].index

        master_mask = (set(maskx)&set(masky))

        return master_mask

    elif len1!=length and len2==length:

        maskx= df[col1_valid].index

        return maskx

    elif len2!=length and len1==length:

         masky= df[col2_valid].index

         return masky
def col_valid_index(col_combo, df):

    """Accepts return of numeric_col and returns dictionary of evey combination in col_combo with a shared index"""

    valid_index_dict = {}

    for combo in col_combo:

        value = mututal_data(combo,df)

        valid_index_dict[combo]=value

    return valid_index_dict
def linreg(combo_dict, df, r=0.1):

    """Takes the return of col_valid_index dictionary, keys are combo tuples, values are range index"""

    dict_lineregress = {}

    for combos in combo_dict:

        xcol,ycol = combos

        x = df.iloc[combo_dict[combos]][xcol]

        y = df.iloc[combo_dict[combos]][ycol]

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        if (abs(r_value)>r):

            dict_lineregress[combos] = stats.linregress(x, y)

    return dict_lineregress
def find_all_linear(df,rvalue=0.1):

    "df is the dataframe, slope is the bare minimum slope defaulted at 0.01"

    numeric_columns_combo = numeric_col(df) #gets ints or floast from a column, when working with linear regression, fields have to be numerical for the scatterplots. returns all combinations of columns

    combo_dict = col_valid_index(numeric_columns_combo, df)#finding the linear regression will return NaN if the columns are NOT equal in length, this function returns a dict of {combos:index in common}

    return linreg(combo_dict, df, r=rvalue) #combo_dict is {(col1,col2):shared range} --> linreg will use stats.linregression and will return dictionary of any two columns with a slope greater than m
find_all_linear(titanic,rvalue=0.3)
x = titanic.Fare

y = titanic.Pclass

fig, ax = plt.subplots(1,1,figsize=(10,10))

sns.regplot(x,y)
family_survival = titanic[(titanic.SibSp>0)|(titanic.Parch>0)].Survived.value_counts()

family_survival
non_family_survival = titanic[(titanic.SibSp==0)|(titanic.Parch==0)].Survived.value_counts()

died = [non_family_survival[0],family_survival[0]]

survived = [non_family_survival[1],family_survival[1]]

observed = [died,survived]

non_family_survival
chi2, p, dof, expected = stats.chi2_contingency(observed, correction=False)
p
# age categories

bins = [0,14,24,64,100]

bins_cat = ['child', 'youth', 'adult', 'senior']

titanic['Age_category'] =pd.cut(titanic.Age, bins=bins, labels=bins_cat)
(titanic.groupby('Age_category').Survived.value_counts(normalize=True).sort_index())
titanic.groupby('Age_category').Survived.value_counts()
#assignning this groupby function to a variable

by_sex = titanic.groupby('Sex')
by_sex.Survived.value_counts(normalize=True).sort_index()
by_sex.Fare.mean()
by_sex.Pclass.value_counts(normalize=True).sort_index()
by_sex.Age.mean()