# Task 1: Data Preparation

# "You will start by loading the CSV data from the file (using appropriate pandas functions) and checking whether the loaded data is equivalent to the data in the source CSV file.

# Then, you need to clean the data by using the knowledge we taught in the lectures. You need to deal with all the potential issues/errors in the data appropriately (such as: typos, extra whitespaces, sanity checks for impossible values, and missing values etc). "



# Please structure code as follows: 

# Always provide one line of comments to explain the purpose of the code, e.g. load the data, checking the equivalent to original data, checking typos (do this for each other types of errors)



#Code goes after this line by adding cells
import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

%matplotlib inline
starwars = pd.read_csv('../input/starwars-survey-in-usa/StarWars.csv',encoding = 'latin1')
starwars.head()
starwars.nunique()
starwars.rename(columns={'Which of the following Star Wars films have you seen? Please select all that apply.' : 'Watched|Movie1',

                        'Unnamed: 4' : 'Watched|Movie2',

                        'Unnamed: 5' : 'Watched|Movie3',

                        'Unnamed: 6' : 'Watched|Movie4',

                        'Unnamed: 7' : 'Watched|Movie5',

                        'Unnamed: 8' : 'Watched|Movie6',

                        'Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.' : 'Rank|Movie1',

                        'Unnamed: 10' : 'Rank|Movie2',

                        'Unnamed: 11' : 'Rank|Movie3',

                        'Unnamed: 12' : 'Rank|Movie4',

                        'Unnamed: 13' : 'Rank|Movie5',

                        'Unnamed: 14' : 'Rank|Movie6',

                        'Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.' : 'Familiarity|Han Solo',

                        'Unnamed: 16' : 'Familiarity|Luke Skywalker',

                        'Unnamed: 17' : 'Familiarity|Princess Leia Organa',

                        'Unnamed: 18' : 'Familiarity|Anakin Skywalker',

                        'Unnamed: 19' : 'Familiarity|Obi Wan Kenobi',

                        'Unnamed: 20' : 'Familiarity|Emperor Palpatine',

                        'Unnamed: 21' : 'Familiarity|Darth Vader',

                        'Unnamed: 22' : 'Familiarity|Lando Calrissian',

                        'Unnamed: 23' : 'Familiarity|Boba Fett',

                        'Unnamed: 24' : 'Familiarity|C-3P0',

                        'Unnamed: 25' : 'Familiarity|R2 D2',

                        'Unnamed: 26' : 'Familiarity|Jar Jar Binks',

                        'Unnamed: 27' : 'Familiarity|Padme Amidala',

                        'Unnamed: 28' : 'Familiarity|Yoda'},inplace = True)
starwars.nunique()
starwars.drop([0],inplace = True)
starwars['Have you seen any of the 6 films in the Star Wars franchise?'].value_counts()
starwars.dtypes
starwars[(starwars.select_dtypes(['object'])).columns] = starwars.select_dtypes(['object']).apply(lambda x: x.str.upper())
starwars['Have you seen any of the 6 films in the Star Wars franchise?'].value_counts()
starwars[(starwars.select_dtypes(['object'])).columns] = starwars.select_dtypes(['object']).apply(lambda x: x.str.strip())
starwars['Have you seen any of the 6 films in the Star Wars franchise?'].value_counts()
for col in starwars.columns[1:]:

    print(starwars[col].value_counts())
starwars['Are you familiar with the Expanded Universe?'].value_counts()
starwars['Do you consider yourself to be a fan of the Expanded Universe?\x8cæ'].value_counts()
typos = {'YESS': 'YES',    

         'NOO':'NO',  

         'F':'FEMALE'}

starwars[(starwars.select_dtypes(['object'])).columns] = starwars.select_dtypes(['object']).apply(lambda x: x.replace(typos))
starwars.rename({'Do you consider yourself to be a fan of the Expanded Universe?\x8cæ' : 'Do you consider yourself to be a fan of the Expanded Universe?'},inplace = True)
starwars.dtypes
for col in starwars.columns[1:]:

    print(starwars[col].value_counts())

    print()    
starwars['Age'].value_counts()
starwars['Age'].replace('500','45-60',inplace = True)
starwars['Age'].value_counts()
for col in 'Watched|Movie1','Watched|Movie2','Watched|Movie3','Watched|Movie4','Watched|Movie5','Watched|Movie6':

    starwars[col] = ~starwars[col].isna()

starwars.head()
starwars.isnull().sum()
lst = []

for index, row in starwars.iterrows():

    if(row[1] == 'YES'):

        if(row[3]* row[4] * row[5] * row[6] * row[7] * row[8]== False):

            lst.append(index)

            

print(len(lst))
starwars = starwars.drop(lst)

starwars = starwars[starwars['Do you consider yourself to be a fan of the Star Wars film franchise?'].notna()]
starwars.shape
starwars.isnull().sum()
ques3 = starwars[starwars.columns[15:]]

ques3.head()
ques3.dropna(inplace = True)
ques3.isna().sum()
# Task 2: Data Exploration

# 1. Explore the survey question: \textit{Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.	(Star Wars: Episode I  The Phantom Menace; Star Wars: Episode II  Attack of the Clones; Star Wars: Episode III  Revenge of the Sith;	Star Wars: Episode IV  A New Hope; Star Wars: Episode V The Empire Strikes Back; Star Wars: Episode VI Return of the Jedi)}, then analysis how people rate Star Wars Movies. 





#Code goes after this line by adding cells
starwars.iloc[:,9:15] = starwars.iloc[:,9:15].astype(float)
cols = ['Rank|Movie1','Rank|Movie2','Rank|Movie3','Rank|Movie4','Rank|Movie5','Rank|Movie6']

avg = {}

data = {}

for col in cols:

    avg[col] = (starwars[col].mean())**-1 #(below for explaination)

data['mean'] = avg

data   

rank = sorted(data['mean'].items())



plt.bar(range(len(data['mean'])), [x[1] for x in rank], align = 'center')

plt.xticks(range(len(data['mean'])), ['Movie1','Movie2','Movie3','Movie4','Movie5','Movie6'])

plt.ylabel('Ranking')

plt.title('Comparison of ranks from Movie 1-6')

plt.show()

plt.xticks
# Task 2: Data Exploration

# 2. Explore the relationships between columns; at least 3 visualisations with plausible hypothesis



#Code goes after this line by adding cells
copy1 = starwars[['Do you consider yourself to be a fan of the Star Wars film franchise?','Age']]
copy1.rename(columns = {'Do you consider yourself to be a fan of the Star Wars film franchise?' : 'outcome'},inplace = True)
copy1.head()
pd.DataFrame(copy1.Age.value_counts())
pd.DataFrame(copy1.outcome.value_counts())
stWars = copy1.groupby("Age").outcome.value_counts(normalize = True)

stWars
stWars.index
stWars.unstack()
plt.figure(figsize=(10,4))



plt.subplot(1,2,1);copy1.outcome.value_counts().plot(kind='bar', color=['C0','C1']); plt.title('Star War Fan')

plt.subplot(1,2,2);copy1.Age.value_counts().plot(kind='bar', color=['C3','C4','C5','C6']); plt.title('Age')
stWars.unstack().plot(kind = 'bar', stacked = True, title = 'Age vs Star War Fans')
copy2 = starwars[['Do you consider yourself to be a fan of the Star Trek franchise?','Age']]
copy2.rename(columns = {'Do you consider yourself to be a fan of the Star Trek franchise?' : 'result'},inplace = True)
copy2.head()
pd.DataFrame(copy2.Age.value_counts())
pd.DataFrame(copy2.result.value_counts())
stTrek = copy2.groupby("Age").result.value_counts(normalize = True)

stTrek
stTrek.index
stTrek.unstack()
plt.figure(figsize=(10,4))



plt.subplot(1,2,1);copy2.result.value_counts().plot(kind='bar', color=['C0','C1']); plt.title('Star Trek Fan')

plt.subplot(1,2,2);copy2.Age.value_counts().plot(kind='bar', color=['C3','C4','C5','C6']); plt.title('Age')
stTrek.unstack().plot(kind = 'bar', stacked = True, title = 'Age vs Star Trek Fans')
copy3 = starwars[['Do you consider yourself to be a fan of the Star Wars film franchise?','Which character shot first?']]



copy3.rename(columns = {'Do you consider yourself to be a fan of the Star Wars film franchise?' : 'fan','Which character shot first?' : 'firstShot'},inplace = True)
copy3.head()
pd.DataFrame(copy3.firstShot.value_counts())
pd.DataFrame(copy3.fan.value_counts())
shot = copy3.groupby("firstShot").fan.value_counts(normalize = True)

shot
type(shot)
shot.unstack()
plt.figure(figsize=(10,4))



plt.subplot(1,2,1);copy3.fan.value_counts().plot(kind='bar', color=['C0','C1']); plt.title('Star War Fan')

plt.subplot(1,2,2);copy3.firstShot.value_counts().plot(kind='bar', color=['C3','C4','C5']); plt.title('Which Charecter Shot First?')
shot.unstack().plot(kind = 'bar', stacked = True, title = 'First Shot vs Star War Fans')

ax = plt.subplot(111)

# Task 2: Data Exploration

# 3. Explore whether there are relationship between people's demographics (Gender, Age, Household Income, Education, Location) and their attitude to Start War characters. 



#Code goes after this line by adding cells
ques3.head() 
gender = ques3[['Familiarity|Han Solo',                                            

'Familiarity|Luke Skywalker',                                          

'Familiarity|Princess Leia Organa',                                    

'Familiarity|Anakin Skywalker',                                        

'Familiarity|Obi Wan Kenobi',                                          

'Familiarity|Emperor Palpatine',                                       

'Familiarity|Darth Vader',                                             

'Familiarity|Lando Calrissian',                                        

'Familiarity|Boba Fett',                                               

'Familiarity|C-3P0',                                                   

'Familiarity|R2 D2',                                                   

'Familiarity|Jar Jar Binks',                                           

'Familiarity|Padme Amidala',                                           

'Familiarity|Yoda',

'Gender']]
gender.head()
gender.nunique()
gender.replace({'UNFAMILIAR (N/A)' : 0,

                'VERY FAVORABLY' : 1,

               'SOMEWHAT FAVORABLY' : 2,

               'NEITHER FAVORABLY NOR UNFAVORABLY (NEUTRAL)' : 3,

               'SOMEWHAT UNFAVORABLY' : 4,

               'VERY UNFAVORABLY' : 5,

               'MALE' : 0,

               'FEMALE' : 1}, inplace = True)
gender.head()
gender.isna().sum()
gender['Familiarity|Emperor Palpatine'].value_counts()
gender.dtypes
gender.groupby('Gender').agg(lambda x:x.value_counts().index[0])
plt.style.use('dark_background')

gender.groupby('Gender').agg(lambda x:x.value_counts().index[0]).plot(kind = 'bar', 

                                                  figsize = (5,5),

                                                 title = 'Charecter Familiarity vs Gender',

                                                 grid = True,

                                                 legend = True,

                                                 colormap = 'plasma')

plt.ylabel('Familiarity')

plt.yticks((1,2,3,4,5),['Very Favourable','Somewhat Favorable','Neutral','Somewhat Unfavorable','Very Unfavorable'])

plt.xticks((0,1),['Male','Female'])

ax = plt.subplot(111)

ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.8), shadow=True, ncol=1)
age = ques3[['Familiarity|Han Solo',                                            

'Familiarity|Luke Skywalker',                                          

'Familiarity|Princess Leia Organa',                                    

'Familiarity|Anakin Skywalker',                                        

'Familiarity|Obi Wan Kenobi',                                          

'Familiarity|Emperor Palpatine',                                       

'Familiarity|Darth Vader',                                             

'Familiarity|Lando Calrissian',                                        

'Familiarity|Boba Fett',                                               

'Familiarity|C-3P0',                                                   

'Familiarity|R2 D2',                                                   

'Familiarity|Jar Jar Binks',                                           

'Familiarity|Padme Amidala',                                           

'Familiarity|Yoda',

'Age']]
age.head()
age['Age'].value_counts()
age.replace({'UNFAMILIAR (N/A)' : 0,

                'VERY FAVORABLY' : 1,

               'SOMEWHAT FAVORABLY' : 2,

               'NEITHER FAVORABLY NOR UNFAVORABLY (NEUTRAL)' : 3,

               'SOMEWHAT UNFAVORABLY' : 4,

               'VERY UNFAVORABLY' : 5,

               '18-29' : 0,

               '30-44' : 1,

            '45-60' : 2,

            '> 60' : 3}, inplace = True)
age.head()
age.dtypes
age.groupby('Age').agg(lambda x:x.value_counts().index[0])
plt.style.use('dark_background')

age.groupby('Age').agg(lambda x:x.value_counts().index[0]).plot(kind = 'bar', 

                                                  figsize = (5,5),

                                                 title = 'Charecter Familiarity vs Age',

                                                 grid = True,

                                                 legend = True,

                                                 colormap = 'plasma')

plt.ylabel('Familiarity')

plt.yticks((1,2,3,4,5),['Very Favourable','Somewhat Favorable','Neutral','Somewhat Unfavorable','Very Unfavorable'])

plt.xticks((0,1,2,3),['18-29','30-44','45-60','Greater than 60'])

ax = plt.subplot(111)

ax.legend(loc='upper center', bbox_to_anchor=(1.4, 0.8), shadow=True, ncol=1)
income = ques3[['Familiarity|Han Solo',                                            

'Familiarity|Luke Skywalker',                                          

'Familiarity|Princess Leia Organa',                                    

'Familiarity|Anakin Skywalker',                                        

'Familiarity|Obi Wan Kenobi',                                          

'Familiarity|Emperor Palpatine',                                       

'Familiarity|Darth Vader',                                             

'Familiarity|Lando Calrissian',                                        

'Familiarity|Boba Fett',                                               

'Familiarity|C-3P0',                                                   

'Familiarity|R2 D2',                                                   

'Familiarity|Jar Jar Binks',                                           

'Familiarity|Padme Amidala',                                           

'Familiarity|Yoda',

'Household Income']]
income.head()
income['Household Income'].value_counts()
income.replace({'UNFAMILIAR (N/A)' : 0,

                'VERY FAVORABLY' : 1,

               'SOMEWHAT FAVORABLY' : 2,

               'NEITHER FAVORABLY NOR UNFAVORABLY (NEUTRAL)' : 3,

               'SOMEWHAT UNFAVORABLY' : 4,

               'VERY UNFAVORABLY' : 5,

               '$0 - $24,999' : 0,

               '$25,000 - $49,999' : 1,

            '$50,000 - $99,999' : 2,

            '$100,000 - $149,999' : 3,

               '$150,000+' : 4}, inplace = True)
income.dtypes
income.groupby('Household Income').agg(lambda x:x.value_counts().index[0])
plt.style.use('dark_background')

income.groupby('Household Income').agg(lambda x:x.value_counts().index[0]).plot(kind = 'bar', 

                                                  figsize = (10,10),

                                                 title = 'Charecter Familiarity vs Income',

                                                 grid = True,

                                                 legend = True,

                                                 colormap = 'plasma')

plt.ylabel('Familiarity')

plt.yticks((1,2,3,4,5),['Very Favourable','Somewhat Favorable','Neutral','Somewhat Unfavorable','Very Unfavorable'])

plt.xticks((0,1,2,3,4),['$0 - $24,999','$25,000 - $49,999','$50,000 - $99,999','$100,000 - $149,999','$150,000+'])

ax = plt.subplot(111)

ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.8), shadow=True, ncol=1)
education = ques3[['Familiarity|Han Solo',                                            

'Familiarity|Luke Skywalker',                                          

'Familiarity|Princess Leia Organa',                                    

'Familiarity|Anakin Skywalker',                                        

'Familiarity|Obi Wan Kenobi',                                          

'Familiarity|Emperor Palpatine',                                       

'Familiarity|Darth Vader',                                             

'Familiarity|Lando Calrissian',                                        

'Familiarity|Boba Fett',                                               

'Familiarity|C-3P0',                                                   

'Familiarity|R2 D2',                                                   

'Familiarity|Jar Jar Binks',                                           

'Familiarity|Padme Amidala',                                           

'Familiarity|Yoda',

'Education']]
education.head()
education['Education'].value_counts()
education.replace({'UNFAMILIAR (N/A)' : 0,

                'VERY FAVORABLY' : 1,

               'SOMEWHAT FAVORABLY' : 2,

               'NEITHER FAVORABLY NOR UNFAVORABLY (NEUTRAL)' : 3,

               'SOMEWHAT UNFAVORABLY' : 4,

               'VERY UNFAVORABLY' : 5,

               'LESS THAN HIGH SCHOOL DEGREE' : 0,

               'HIGH SCHOOL DEGREE' : 1,

            'SOME COLLEGE OR ASSOCIATE DEGREE' : 2,

            'BACHELOR DEGREE' : 3,

               'GRADUATE DEGREE' : 4}, inplace = True)
education.dtypes
education.groupby('Education').agg(lambda x:x.value_counts().index[0])
plt.style.use('dark_background')

education.groupby('Education').agg(lambda x:x.value_counts().index[0]).plot(kind = 'bar', 

                                                  figsize = (15,10),

                                                 title = 'Charecter Familiarity vs Education level',

                                                 grid = True,

                                                 legend = True,

                                                 colormap = 'plasma')

plt.ylabel('Familiarity')

plt.yticks((1,2,3,4,5),['Very Favourable','Somewhat Favorable','Neutral','Somewhat Unfavorable','Very Unfavorable'])

plt.xticks((0,1,2,3,4),['LESS THAN HIGH SCHOOL DEGREE','HIGH SCHOOL DEGREE','SOME COLLEGE OR ASSOCIATE DEGREE','BACHELOR DEGREE','GRADUATE DEGREE'])

ax = plt.subplot(111)

ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.8), shadow=True, ncol=1)
location = ques3[['Familiarity|Han Solo',                                            

'Familiarity|Luke Skywalker',                                          

'Familiarity|Princess Leia Organa',                                    

'Familiarity|Anakin Skywalker',                                        

'Familiarity|Obi Wan Kenobi',                                          

'Familiarity|Emperor Palpatine',                                       

'Familiarity|Darth Vader',                                             

'Familiarity|Lando Calrissian',                                        

'Familiarity|Boba Fett',                                               

'Familiarity|C-3P0',                                                   

'Familiarity|R2 D2',                                                   

'Familiarity|Jar Jar Binks',                                           

'Familiarity|Padme Amidala',                                           

'Familiarity|Yoda',

'Location (Census Region)']]
location.head()
location['Location (Census Region)'].value_counts()
location.replace({'UNFAMILIAR (N/A)' : 0,

                'VERY FAVORABLY' : 1,

               'SOMEWHAT FAVORABLY' : 2,

               'NEITHER FAVORABLY NOR UNFAVORABLY (NEUTRAL)' : 3,

               'SOMEWHAT UNFAVORABLY' : 4,

               'VERY UNFAVORABLY' : 5,

               'SOUTH ATLANTIC' : 0,

               'PACIFIC' : 1,

            'EAST NORTH CENTRAL' : 2,

            'WEST SOUTH CENTRAL' : 3,

               'MIDDLE ATLANTIC' : 4,

                 'NEW ENGLAND' : 5,

                 'WEST NORTH CENTRAL' : 6,

                 'MOUNTAIN' : 7,

                 'EAST SOUTH CENTRAL' : 8}, inplace = True)
location.dtypes
location.groupby('Location (Census Region)').agg(lambda x:x.value_counts().index[0])
plt.style.use('dark_background')

location.groupby('Location (Census Region)').agg(lambda x:x.value_counts().index[0]).plot(kind = 'bar', 

                                                  figsize = (17,10),

                                                 title = 'Charecter Familiarity vs Location',

                                                 grid = True,

                                                 legend = True,

                                                 colormap = 'plasma')

plt.ylabel('Familiarity')

plt.yticks((1,2,3,4,5),['Very Favourable','Somewhat Favorable','Neutral','Somewhat Unfavorable','Very Unfavorable'])

plt.xticks((0,1,2,3,4,5,6,7,8),['SOUTH ATLANTIC','PACIFIC','EAST NORTH CENTRAL','WEST SOUTH CENTRAL','MIDDLE ATLANTIC','NEW ENGLAND','WEST NORTH CENTRAL','MOUNTAIN','EAST SOUTH CENTRAL'])

ax = plt.subplot(111)

ax.legend(loc='upper center', bbox_to_anchor=(1.10, 0.8), shadow=True, ncol=1)