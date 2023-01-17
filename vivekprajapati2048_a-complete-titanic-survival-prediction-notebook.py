# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np  # linear algebra

import pandas as pd  # data processing



# data visualization for EDA

%matplotlib inline

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns



import string



# ignore warnings

import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")



print(df_train.shape)

print(df_test.shape)
# concatenate data of training and test set

def concat_df(train_data, test_data):

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)





df_all = concat_df(df_train, df_test)
# divided data of training and test set

def divide_df(all_data):

    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis = 1)
df_all.sample(10)  # generate a sample random row and column
df_train.info()
df_train.describe()
f, ax = plt.subplots(1, 2, figsize=(18,8))  # 1 x 2 subplots



survived = df_train['Survived'].value_counts()

survived.plot.pie(explode=[0, 0.1], 

                  autopct='%1.1f%%', 

                  ax=ax[0], 

                  shadow=True)



ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('Survived')



plt.show()
missing_values_train = df_train.isnull().sum()  # number of missing values



missing_values_train.sort_values(ascending=False)
# finding percentage of missing values



missing_percent = 100 * missing_values_train / len(df_train)

missing_percent_train = (round(missing_percent, 1))  # rounding off



missing_percent_train.sort_values(ascending=False)
# creating dataframe of missing values



df_missing_train = pd.concat([missing_values_train, 

                              missing_percent_train], axis=1, keys=['Total', '%'])

df_missing_train.sort_values(by='%', ascending=False)
missing_values_test = df_test.isnull().sum()  # number of missing values



missing_values_test.sort_values(ascending=False)
# finding percentage of missing values



missing_percent = 100 * missing_values_test / len(df_test)

missing_percent_test = (round(missing_percent, 1))  # rounding off



missing_percent_test.sort_values(ascending=False)
# creating dataframe of missing values



df_missing_test = pd.concat([missing_values_test, 

                             missing_percent_test], axis=1, keys=['Total', '%'])

df_missing_test.sort_values(by='%', ascending=False)
f, ax = plt.subplots(figsize=(18,8))



sns.violinplot("Pclass", "Age", 

               hue="Survived", 

               data=df_train, 

               split=True, 

               ax=ax)



ax.set_title('Pclass and Age vs Survived')

ax.set_yticks(range(0, 110, 10))



plt.show()
df_all_corr = df_all.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()



df_all_corr.rename(columns={"level_0": "Feature 1",

                            "level_1": "Feature 2",

                            0: "Correlation Coefficient"},

                            inplace = True)

df_all_corr[df_all_corr['Feature 1'] == 'Pclass']
f, ax = plt.subplots(figsize=(18,8))



sns.violinplot("Sex", "Age", hue="Survived", data=df_train, split=True, ax=ax)

ax.set_title('Sex and Age vs Survived')

ax.set_yticks(range(0, 110, 10))



plt.show()
age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']



for pclass in range(1,4):

    for sex in ['female', 'male']:

        print('Median age of Pclass {} {}s: {}'.format(pclass, 

                sex, age_by_pclass_sex[sex][pclass].astype(int)))

        

df_all['Age'] = df_all.groupby(['Sex','Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
sns.factorplot('Embarked', 'Survived', data=df_train)

fig = plt.gcf()  # used to get current figure

fig.set_size_inches(5,3)

plt.show()
df_all[df_all['Embarked'].isnull()]
df_all['Embarked'] = df_all['Embarked'].fillna('S')
FaceGrid = sns.FacetGrid(df_train, row='Embarked', size=3.5, aspect=1.6)

FaceGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')

FaceGrid.add_legend()
df_all[df_all['Fare'].isnull()]
med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].median()[3][0][0]

med_fare  # median of Fare satisfying condition([3][0][0] -- 3=Pclass, 0=Parch, SibSp)
df_all['Fare'] = df_all['Fare'].fillna(med_fare)
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df_train)
grid = sns.FacetGrid(df_train, col='Survived', 

                    row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=0.5, bins=20)

grid.add_legend();
data1 = df_train.copy()  # shallow  copy

data1['family_size'] = data1['SibSp'] + data1['Parch'] + 1 # 1 if a person is alone

data1['family_size'].value_counts().sort_values(ascending=False)
axes = sns.factorplot('family_size', 'Survived', data=data1, aspect=2.5)
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')



df_all_decks = df_all.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 

                                                                        'SibSp', 'Parch', 'Fare', 'Embarked', 

                                                                        'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name':'Count'})

# doing transpose for accessibility

df_all_decks = df_all_decks.transpose()



df_all_decks
def get_pclass_dist(df):

    

    # create a dictionary of every deck for 'passenger count' in every class

    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 

                   'F': {}, 'G': {}, 'M': {}, 'T': {}}

    

    # extract deck column from df_all_decks

    decks = df.columns.levels[0]

    

    # create a new dataframe with '0' if empty in respective 'Pclass' of df_all_decks

    for deck in decks:

        for pclass in range(1,4):

            try:

                count = df[deck][pclass][0]

                deck_counts[deck][pclass] = count

            except KeyError:

                deck_counts[deck][pclass] = 0

    

    df_decks = pd.DataFrame(deck_counts)

    

    

    

    # create a dictionary of every deck for 'percentage count' of passangers in every class 

    deck_percentages = {}

    

    for col in df_decks.columns:

        deck_percentages[col] = [(count/df_decks[col].sum()) * 100 

                                 for count in df_decks[col]]

    

    return deck_counts, df_decks, deck_percentages







all_deck_count, df_decks_return, all_deck_percent = get_pclass_dist(df_all_decks)
print(df_decks_return)  # returns a dataframe of passenger count

all_deck_count  # returns a dictionary of passenger count in every class
all_deck_percent # returns a percentage count of passengers
def display_pclass_dist(percentages):

    

    # converting dictionary into dataframe and then transpose

    df_percentages = pd.DataFrame(percentages).transpose()

    

    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')

    bar_count = np.arange(len(deck_names))

    bar_width = 0.85

    

    pclass1 = df_percentages[0]

    pclass2 = df_percentages[1]

    pclass3 = df_percentages[2]

    

    plt.figure(figsize=(20,10))

    

    plt.bar(bar_count, pclass1, color='brown',

            edgecolor='white', width=bar_width, label='Passenger Class 1')

    plt.bar(bar_count, pclass2, bottom=pclass1, color='teal', 

            edgecolor='white', width=bar_width, label='Passenger Class 2')

    plt.bar(bar_count, pclass3, bottom=pclass1+pclass2, color='peru', 

            edgecolor='white', width=bar_width, label='Passenger Class 3')

    

    plt.xlabel('Deck', size=15, labelpad=20)

    plt.ylabel('Passenger Class Percentage', size=15, labelpad=20)

    

    plt.xticks(bar_count, deck_names)

    

    plt.tick_params(axis='x', labelsize=15)

    plt.tick_params(axis='y', labelsize=15)

    

    plt.legend(loc='best', bbox_to_anchor=(1,1), prop={'size':15})

    plt.title('Passenger Class Distribution in Decks', size=18, y=1.05)

    

    plt.show()

    

display_pclass_dist(all_deck_percent)
idx = df_all[df_all['Deck'] == 'T'].index

df_all.loc[idx, 'Deck'] = 'A'    
df_all_survived = df_all.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 

                                                                             'Fare', 'Embarked', 'Pclass', 

                                                                             'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name':'Count'}).transpose()
def get_survived_dist(df):

    

    # create a dictionary for 'survival count' in every deck

    survival_counts = {'A':{}, 'B':{}, 'C':{}, 'D':{}, 

                       'E':{}, 'F':{}, 'G':{}, 'M':{}}

    

    # extract deck column from df_all_decks

    decks = df.columns.levels[0]

    

    

    for deck in decks:

        for survive in range(0,2):

            survival_counts[deck][survive] = df[deck][survive][0]

            

    df_survival = pd.DataFrame(survival_counts)

    

    # create a dictionary of 'survival count' in every class

    survival_percentages = {}

    

    for col in df_survival.columns:

        survival_percentages[col] = [(count/df_survival[col].sum()) * 100 

                                     for count in df_survival[col]]

    

    return survival_counts, survival_percentages



all_survival_count, all_survival_percentage = get_survived_dist(df_all_survived)
all_survival_count
all_survival_percentage
def display_survival_dist(percentages):

    

    df_survival_percentages = pd.DataFrame(percentages).transpose()

    

    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')

    bar_count = np.arange(len(deck_names))

    bar_width = 0.85

    

    not_survived = df_survival_percentages[0]

    survived = df_survival_percentages[1]

    

    plt.figure(figsize=(20,10))

    plt.bar(bar_count, not_survived, color='grey', 

            edgecolor='white', width=bar_width, label='Not Survived')

    plt.bar(bar_count, survived, color='deepskyblue', bottom=not_survived, 

            edgecolor='white', width=bar_width, label="Survived")

    

    plt.xlabel('Deck', size=15, labelpad=20)

    plt.ylabel('Survival Percentage', size=15, labelpad=20)

    plt.xticks(bar_count, deck_names)

    plt.tick_params(axis='x', labelsize=15)

    plt.tick_params(axis='y', labelsize=15)

    

    plt.legend(loc='upper left', bbox_to_anchor=(1,1), prop={'size':15})

    plt.title('Survival Percentage in Decks', size=18, y=1.05)

    

    plt.show()

    

display_survival_dist(all_survival_percentage)
df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')

df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')

df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')



df_all['Deck'].value_counts()
df_all.drop(['Cabin'], inplace=True, axis=1)



df_all.head()
df_train, df_test = divide_df(df_all)

dfs = [df_train, df_test]



for df in dfs:

    print(df.isnull().sum())

    print("\n")
continuous_features = ['Age', 'Fare']

survived = df_train['Survived'] == 1



fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))

plt.subplots_adjust(right=1.5)



for i, feature in enumerate(continuous_features):

    

    # distribution of survival in feature

    sns.distplot(df_train[~survived][feature], label = 'Not Survived', 

                 hist=True, color='teal', ax=axs[0][i]) 

                        # [~survived] means 'Not Survived'

    sns.distplot(df_train[survived][feature], label='Survived', 

                 hist=True, color='peru', ax=axs[0][i])

    

    # distribution of feature in dataset

    sns.distplot(df_train[feature], label='Training Set', 

                 hist=False, color='teal', ax=axs[1][i])

    sns.distplot(df_test[feature], label='Test Set', 

                 hist=False, color='peru', ax=axs[1][i])

    

    axs[0][i].set_xlabel('')

    axs[1][i].set_xlabel('')

    

    # providing the ticks for x and y in respective plots

    for j in range(2):

        axs[i][j].tick_params(axis='x', labelsize=20)

        axs[i][j].tick_params(axis='y', labelsize=20)

        

    axs[0][i].legend(loc='upper right', prop={'size': 20})

    axs[1][i].legend(loc='upper right', prop={'size': 20})

    

    axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=20, y=1.05)



axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=20, y=1.05)

axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=20, y=1.05)

    

plt.show()
categorical_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Deck']



fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20,20))

plt.subplots_adjust(right=1.5, top=1.25)



for i, feature in enumerate(categorical_features, 1):

    plt.subplot(2, 3, i)

    sns.countplot(x=feature, hue='Survived', data=df_train)

    

    plt.xlabel('{}'.format(feature), size=20, labelpad=15)

    plt.ylabel('Passenger Count', size=20, labelpad=15)

    

    plt.tick_params(axis='x', labelsize=20)

    plt.tick_params(axis='y', labelsize=20)

    

    plt.legend(['Not Survived', 'Survived'], loc='upper centre', prop={'size':18})

    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)



plt.show()
sns.heatmap(df_all.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)



fig = plt.gcf()

fig.set_size_inches(10,8)



plt.show()
df_all['Fare'] = pd.qcut(df_all['Fare'], 13)
fig, axs = plt.subplots(figsize=(22, 9))

sns.countplot(x='Fare', hue='Survived', data=df_all)



plt.xlabel('Fare', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)



plt.tick_params(axis='x', labelsize=10)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size':15})

plt.title('Count of Survival in {} feature'.format('Fare'), size=15, y=1.05)



plt.show()
df_all['Age'] = pd.qcut(df_all['Age'], 10)
fig, axs = plt.subplots(figsize=(22, 9))

sns.countplot(x='Age', hue='Survived', data=df_all)



plt.xlabel('Age', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)



plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size':15})

plt.title('Count of Survival in {} feature'.format('Age'), size=15, y=1.05)



plt.show()
df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1



fig, axs = plt.subplots(figsize=(20,20), ncols=2, nrows=2)

plt.subplots_adjust(right=1.5)



sns.barplot(x=df_all['Family_Size'].value_counts().index, 

            y=df_all['Family_Size'].value_counts().values, 

            ax=axs[0][0])



sns.countplot(x='Family_Size', hue='Survived', 

              data=df_all, ax=axs[0][1])



axs[0][0].set_title('Family Size Feature Value Counts', size=20, y=1.05)

axs[0][1].set_title('Survival Counts in Family Size', size=20, y=1.05)





# mapping family size

family_map = {1:'Alone', 

              2:'Small', 3:'Small', 4:'Small', 

              5:'Medium', 6:'Medium', 

              7:'Large', 8:'Large', 11:'Large'}



df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)



sns.barplot(x=df_all['Family_Size_Grouped'].value_counts().index, 

            y=df_all['Family_Size_Grouped'].value_counts().values, 

            ax=axs[1][0])



sns.countplot(x='Family_Size_Grouped', hue='Survived', 

              data=df_all, ax=axs[1][1])



axs[1][0].set_title('Family Size Feature Value Counts After Grouping', size=20, y=1.05)

axs[1][1].set_title('Survival Counts in Family Size After Grouping', size=20, y=1.05)



for i in range(2):

    axs[i][1].legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 20})

    for j in range(2):

        axs[i][j].tick_params(axis='x', labelsize=20)

        axs[i][j].tick_params(axis='y', labelsize=20)

        axs[i][j].set_xlabel('')

        axs[i][j].set_ylabel('')

        

plt.show()
df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')
fig, axs = plt.subplots(figsize=(12,9))

sns.countplot(x='Ticket_Frequency', hue='Survived', data=df_all)



plt.xlabel('Ticket Frequency', size = 15, labelpad=20)

plt.ylabel('Passenger Count', size = 15, labelpad=20)



plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size':15})

plt.title('Count of Survival in {} Feature'.format('Ticket Frequency'), size=15, y=1.05)



plt.show()
df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]



df_all['Is_Married'] = 0

df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1
fig, axs = plt.subplots(nrows=2, figsize=(20,20))

sns.barplot(x=df_all['Title'].value_counts().index, 

            y=df_all['Title'].value_counts().values, 

            ax=axs[0])



axs[0].tick_params(axis='x', labelsize=10)

axs[1].tick_params(axis='y', labelsize=15)



for i in range(2):

    axs[i].tick_params(axis='y', labelsize=15)

    

axs[0].set_title('Title Feature Value Counts', size=20, y=1.05)



df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs', 'Ms', 

                                           'Mlle', 'Lady', 'Mme', 

                                           'the Countess', 'Dona'], 'Miss/Mrs/Ms')

df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 

                                           'Jonkheer', 'Capt', 'Sir', 

                                           'Don', 'Rev'], 'Dr/Military/Noble/Clergy')



sns.barplot(x=df_all['Title'].value_counts().index, 

            y=df_all['Title'].value_counts().values, 

            ax=axs[1])

axs[1].set_title('Title Feature Value Counts After Grouping', size=20, y=1.05)



plt.show()
df_train, df_test = divide_df(df_all)

dfs = [df_train, df_test]
def extract_surname(data):

    

    families = []

    

    for i in range(len(data)):

        name = data.iloc[i]

        

        if '(' in name:

            name_no_bracket = name.split('(')[0]

        else:

            name_no_bracket = name

            

        family = name_no_bracket.split(',')[0]

        title = name_no_bracket.split(',')[1].strip().split(' ')[0]

        

        for c in string.punctuation:

            family = family.replace(c, ' ').strip()

            

        families.append(family)

        

    return families
df_all['Family'] = extract_surname(df_all['Name'])

df_train = df_all.loc[:890]

df_test = df_all.loc[891:]

dfs = [df_train, df_test]
# Creating a list of families and tickets that are occuring in both training and test set

non_unique_families = [x for x in df_train['Family'].unique() if x in df_test['Family'].unique()]

non_unique_tickets = [x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]



df_family_survival_rate = df_train.groupby('Family')['Survived', 'Family', 'Family_Size'].median()

df_ticket_survival_rate = df_train.groupby('Ticket')['Survived', 'Ticket', 'Ticket_Frequency'].median()



family_rates = {}

ticket_rates = {}



# Checking a family exist in both training and test set and has members more than 1

for i in range(len(df_family_survival_rate)):

    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i,1] > 1:

        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i,0]

        

# Checking a ticket exists in both training and test set and has members more than 1

for i in range(len(df_ticket_survival_rate)):

    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i,1] > 1:

        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i,0]
mean_survival_rate = np.mean(df_train['Survived'])



train_family_survival_rate = []

train_family_survival_rate_NA = []



test_family_survival_rate = []

test_family_survival_rate_NA = []





for i in range(len(df_train)):

    if df_train['Family'][i] in family_rates:

        train_family_survival_rate.append(family_rates[df_train['Family'][i]])

        train_family_survival_rate_NA.append(1)

    else:

        train_family_survival_rate.append(mean_survival_rate)

        train_family_survival_rate_NA.append(0)     



for i in range(len(df_test)):

    if df_test['Family'].iloc[i] in family_rates:

        test_family_survival_rate.append(family_rates[df_test['Family'].iloc[i]])

        test_family_survival_rate_NA.append(1)

    else:

        test_family_survival_rate.append(mean_survival_rate)

        test_family_survival_rate_NA.append(0)

        



df_train['Family_Survival_Rate'] = train_family_survival_rate

df_train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA



df_test['Family_Survival_Rate'] = test_family_survival_rate

df_test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA  
train_ticket_survival_rate = []

train_ticket_survival_rate_NA = []



test_ticket_survival_rate = []

test_ticket_survival_rate_NA = []





for i in range(len(df_train)):

    if df_train['Ticket'][i] in ticket_rates:

        train_ticket_survival_rate.append(ticket_rates[df_train['Ticket'][i]])

        train_ticket_survival_rate_NA.append(1)

    else:

        train_ticket_survival_rate.append(mean_survival_rate)

        train_ticket_survival_rate_NA.append(0)     



for i in range(len(df_test)):

    if df_test['Ticket'].iloc[i] in ticket_rates:

        test_ticket_survival_rate.append(ticket_rates[df_test['Ticket'].iloc[i]])

        test_ticket_survival_rate_NA.append(1)

    else:

        test_ticket_survival_rate.append(mean_survival_rate)

        test_ticket_survival_rate_NA.append(0)

        



df_train['Ticket_Survival_Rate'] = train_ticket_survival_rate

df_train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA



df_test['Ticket_Survival_Rate'] = test_ticket_survival_rate

df_test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA  
for df in [df_train, df_test]:

    df['Survival_Rate'] = (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2

    df['Survival_Rate_NA'] = (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']



for df in dfs:

    for feature in non_numeric_features:

        df[feature] = LabelEncoder().fit_transform(df[feature])
onehot_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Grouped']

encoded_features = []



for df in dfs:

    for feature in onehot_features:

        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1,1)).toarray()

        n = df[feature].nunique()

        cols = ['{}_{}'.format(feature, n) for n in range(1,n+1)]

        encoded_df = pd.DataFrame(encoded_feat, columns=cols)

        encoded_df.index = df.index

        encoded_features.append(encoded_df)
# *encoded_features will give all encoded features of each of the Six onehot_features



df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)

df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)
df_all = concat_df(df_train, df_test)



# Dropping unwanted features

drop_cols = ['Deck', 'Embarked', 'Family', 'Family_Size', 'Family_Size_Grouped', 

             'Survived', 'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 

             'Ticket', 'Title', 'Ticket_Survival_Rate', 'Family_Survival_Rate', 

             'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA']



df_all.drop(columns=drop_cols, inplace=True)

df_all.head()
# model helpers

from sklearn.preprocessing import StandardScaler



# models

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB
X_train = StandardScaler().fit_transform(df_train.drop(columns=drop_cols))

y_train = df_train['Survived'].values



print('X_train shape: {}'.format(X_train.shape))

print('Y_train shape: {}'.format(y_train.shape))





X_test = StandardScaler().fit_transform(df_test.drop(columns=drop_cols))

print('X_test shape: {}'.format(X_test.shape))
# Stochastic Gradient Descent (SGD)

sgd = SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

sgd_score = ('{:.2f}'.format(sgd.score(X_train, y_train)*100))





# Random Forest

random_forest = RandomForestClassifier(n_estimators = 100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

random_forest_score = ('{:.2f}'.format(random_forest.score(X_train, y_train)*100))





# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

logreg_score = ('{:.2f}'.format(logreg.score(X_train, y_train)*100))





# K Nearest Neighbor

knn= KNeighborsClassifier(n_neighbors =3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

knn_score = ('{:.2f}'.format(knn.score(X_train, y_train)*100))





# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

y_pred = gaussian.predict(X_test)

gaussian_score = ('{:.2f}'.format(gaussian.score(X_train, y_train)*100))





# Perceptron

perceptron = Perceptron(max_iter = 5)

perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)

perceptron_score = ('{:.2f}'.format(perceptron.score(X_train, y_train)*100))





# Linear Support Vector Machine

linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

y_pred = linear_svc.predict(X_test)

linear_svc_score = ('{:.2f}'.format(linear_svc.score(X_train, y_train)*100))





# Decision Tree

dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

dt_score = ('{:.2f}'.format(dt.score(X_train, y_train)*100))
results = pd.DataFrame({'Model': ['Stochastic Gradient Descent', 'Random Forest', 

                                  'Logistic Regression', 'K Nearest Neighbor', 'Naive Bayes', 

                                  'Perceptron', 'Support Vector Machine', 'Decision Tree'], 

                        'Score': [sgd_score, random_forest_score, logreg_score, knn_score, 

                                  gaussian_score, perceptron_score, linear_svc_score, dt_score]})



df_results = results.sort_values(by='Score', ascending=False)

df_results = df_results.set_index('Score')

df_results
from sklearn.model_selection import cross_val_score



rf = RandomForestClassifier(n_estimators=100, oob_score=True)

scores = cross_val_score(rf, X_train, y_train, cv=10, scoring='accuracy')



print('Scores:', scores)

print('Mean:', scores.mean())

print('Standard Deviation:', scores.std())
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rf.score(X_train, y_train)



rf_score = ('{:.2f} %'.format(rf.score(X_train, y_train)*100))

rf_score
X = df_train.drop(columns=drop_cols)



importances = pd.DataFrame({'feature': X.columns, 

                            'importance': np.round(rf.feature_importances_, 3)})

importances = importances.sort_values('importance', ascending=False).set_index('feature')





importances.head(26)
importances.plot.bar()
random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)



random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)



random_forest_score = ('{:.2f} %'.format(random_forest.score(X_train, y_train)*100))

random_forest_score
print(" random forest oob score:", round(random_forest.oob_score_, 4)*100, "%")
print("rf oob score:", round(rf.oob_score_, 4)*100, "%")
random_forest = RandomForestClassifier(criterion='gini', 

                                           n_estimators=1100,

                                           max_depth=5,

                                           min_samples_split=4,

                                           min_samples_leaf=5,

                                           max_features='auto',

                                           oob_score=True,

                                           random_state=42,

                                           n_jobs=-1,

                                           verbose=1)

random_forest.fit(X_train, y_train)

y_pred = (random_forest.predict(X_test)).astype(int)



random_forest.score(X_train, y_train)

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
from sklearn.model_selection import cross_val_score



rf = RandomForestClassifier(n_estimators=100, oob_score=True)

scores = cross_val_score(random_forest, X_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
from sklearn.model_selection import cross_val_predict



pred = cross_val_predict(random_forest, X_train, y_train, cv=3)
from sklearn.metrics import confusion_matrix

print('Confusion_Matrix:\n {}'.format(confusion_matrix(y_train, pred)))
from sklearn.metrics import precision_score, recall_score

print('Precision:', precision_score(y_train, pred))

print('Recall:', recall_score(y_train, pred))
from sklearn.metrics import f1_score

print('F1 Score:', f1_score(y_train, pred))
from sklearn.metrics import roc_curve



# Getting probabilities of our predictions

y_scores = random_forest.predict_proba(X_train)

y_scores = y_scores[:,1]



# Computing true positive rate and false positive rate

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)



# Plotting them against each other

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):

    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)

    plt.plot([0,1], [0,1], 'r', linewidth=4)

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate (FPR)', fontsize=16)

    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

    

plt.figure(figsize=(14,7))

plot_roc_curve(false_positive_rate, true_positive_rate)

plt.show()
from sklearn.metrics import roc_auc_score



r_a_score = roc_auc_score(y_train, y_scores) 

print('ROC-AUC-Score:', r_a_score)
submission  = pd.DataFrame({'PassengerId': df_test['PassengerId'], 

                            'Survived': y_pred})



submission.to_csv('my_submission.csv', index=False)
