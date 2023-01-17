import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#%matplotlib inline

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train.head()



def concat_df(train_data, test_data):

    # Returns a concatenated df of training and test set on axis 0

    return pd.concat([train_data, test_data]).reset_index(drop=True)



df_all = concat_df(df_train, df_test)



df_train.name = 'Training Set'

df_test.name = 'Test Set'

df_all.name = 'All Set' 



dfs = [df_train, df_test]



print('Number of Training Examples = {}'.format(df_train.shape[0]))

print('Number of Test Examples = {}\n'.format(df_test.shape[0]))

print('Training X Shape = {}'.format(df_train.shape))

print('Training y Shape = {}\n'.format(df_train['Survived'].shape[0]))

print('Test X Shape = {}'.format(df_test.shape))

print('Test y Shape = {}\n'.format(df_test.shape[0]))

print(df_train.columns)

print(df_test.columns)



#Below function is for trying if this strategy works, but it doesn't work standalone, we tried to further decipher from the generated code.

def fill_cabin_of_spouse(dset):

    for i in range(len(dset)): 

        name = str(dset.loc[i, "Name"])

        cab = str(dset.loc[i, "Cabin"])

        names = name.split(',');

        lastname = names[0]

        if (cab == "nan"):

            for j in range(len(dset)):

                if (j == i):

                    continue

                else:

                    name2 = str(dset.loc[j, "Name"])

                    names2 = name2.split(',');

                    lastname2 = names2[0]

                    if (lastname2 == lastname):

                        dset.loc[i, "Cabin"] = dset.loc[j, "Cabin"]

                        break

    return dset



#Filling the mising value of cabin with the value of Spouse's

df_all = fill_cabin_of_spouse(df_all)

#df_all['Cabin'] = df_all['Cabin'].fillna("Unknown")



age_by_class_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']



for pclass in range(1, 4):

    for sex in ['female', 'male']:

        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_class_sex[sex][pclass]))

print('Median age of all passengers: {}'.format(df_all['Age'].median()))



# Filling the missing values in Age with the medians of Sex and Pclass attributes

df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))





# Creating Deck column from the first letter of the Cabin column (M stands for Missing)

df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')



df_all_decks = df_all.groupby(['Deck', 'Pclass']).count().drop(['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin', 'PassengerId', 'Ticket'], axis=1).rename({'Name': 'Count'}).transpose()



def get_pclass_dist(df):

    

    # Creating a dictionary for every passenger class count in every deck

    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}

    decks = df.columns.levels[0]    

    

    for deck in decks:

        for pclass in range(1, 4):

            try:

                count = df[deck][pclass][0]

                deck_counts[deck][pclass] = count 

            except KeyError:

                deck_counts[deck][pclass] = 0

                

    df_decks = pd.DataFrame(deck_counts)    

    deck_percentages = {}



    # Creating a dictionary for every passenger class percentage in every deck

    for col in df_decks.columns:

        deck_percentages[col] = [(count / df_decks[col].sum()) * 100 for count in df_decks[col]]

        

    return deck_counts, deck_percentages



def display_pclass_dist(percentages):

    

    df_percentages = pd.DataFrame(percentages).transpose()

    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')

    bar_count = np.arange(len(deck_names))  

    bar_width = 0.50

    

    pclass1 = df_percentages[0]

    pclass2 = df_percentages[1]

    pclass3 = df_percentages[2]

    

    plt.figure(figsize=(20, 10))

    plt.bar(bar_count, pclass1, color='#ff726f', edgecolor='white', width=bar_width, label='Lower')

    plt.bar(bar_count, pclass2, bottom=pclass1, color='#add8e6', edgecolor='white', width=bar_width, label='Middle')

    plt.bar(bar_count, pclass3, bottom=pclass1 + pclass2, color='#90ee90', edgecolor='white', width=bar_width, label='Upper')



    plt.xlabel('Deck', size=12)

    plt.ylabel('Passenger Class Percentage', size=12)

    plt.xticks(bar_count, deck_names)    

    plt.tick_params(axis='x', labelsize=15)

    plt.tick_params(axis='y', labelsize=15)

    

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 10})

    plt.title('Passenger Class Distribution in Decks', size=12, y=1.05)   

    

    plt.show()    



print(df_all_decks.head())

deck_count, deck_percent = get_pclass_dist(df_all_decks)

display_pclass_dist(deck_percent)



#idx = df_all[df_all['Deck'] == 'T'].index

#df_all.loc[idx, 'Deck'] = 'A'



df_all_decks_survived = df_all.groupby(['Deck', 'Survived']).count().drop(['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket'],axis=1).rename({'Name': 'Count'}).transpose()



def get_survived_dist(df):

    

    # Creating a dictionary for every survival count in every deck

    surv_counts = {'A':{}, 'B':{}, 'C':{}, 'D':{}, 'E':{}, 'F':{}, 'G':{}, 'M':{}, 'T':{}}

    decks = df.columns.levels[0]    



    for deck in decks:

        for survive in range(0, 2):

            if (survive in df[deck]):

                surv_counts[deck][survive] = df[deck][survive][0]

            

    df_surv = pd.DataFrame(surv_counts)

    surv_percentages = {}



    for col in df_surv.columns:

        surv_percentages[col] = [(count / df_surv[col].sum()) * 100 for count in df_surv[col]]

        

    return surv_counts, surv_percentages



def display_surv_dist(percentages):

    

    df_survived_percentages = pd.DataFrame(percentages).transpose()

    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')

    bar_count = np.arange(len(deck_names))  

    bar_width = 0.50  



    not_survived = df_survived_percentages[0.0]

    survived = df_survived_percentages[1.0]

    

    plt.figure(figsize=(20, 10))

    plt.bar(bar_count, not_survived, color='#b5ffb9', edgecolor='white', width=bar_width, label="Not Survived")

    plt.bar(bar_count, survived, bottom=not_survived, color='#f9bc86', edgecolor='white', width=bar_width, label="Survived")

 

    plt.xlabel('Deck', size=15, labelpad=20)

    plt.ylabel('Survival Percentage', size=15, labelpad=20)

    plt.xticks(bar_count, deck_names)    

    plt.tick_params(axis='x', labelsize=15)

    plt.tick_params(axis='y', labelsize=15)

    

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})

    plt.title('Survival Percentage in Decks', size=18, y=1.05)

    

    plt.show()



print(df_all_decks_survived.head())

all_surv_count, all_surv_per = get_survived_dist(df_all_decks_survived)

display_surv_dist(all_surv_per)



#Grouped the similar decks from the distribution graph, leaving M and T as such

df_all['Deck'] = df_all['Deck'].replace(['A', 'G'], 'AG')

df_all['Deck'] = df_all['Deck'].replace(['B', 'C', 'F'], 'BCF')

df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')





df_all['Deck'].value_counts()



#now we can drop the cabin attribute

df_all.drop(['Cabin'], inplace=True, axis=1)



df_all.head()