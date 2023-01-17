# Data analysis packages:
import pandas as pd
import numpy as np

# Visualization packages:
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from math import pi

%matplotlib inline
## Forcing pandas to display any number of elements
pd.set_option('display.max_columns', None)
pd.options.display.max_seq_items = 2000
## Reading the data:
hackerRank_codebook = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Codebook.csv')
hackerRank_numericMapping = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Numeric-Mapping.csv')
hackerRank_numeric = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Numeric.csv')
hackerRank_values = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Values.csv')
## Selecting the female respondents:
dataset = hackerRank_values[hackerRank_values['q3Gender'] == 'Female']
dataset.info()
## Attributes of interest:
attributes = ['q1AgeBeginCoding', 'q2Age', 'q3Gender', 'q4Education', 'q0004_other',
       'q5DegreeFocus', 'q0005_other', 'q6LearnCodeUni',
       'q6LearnCodeSelfTaught', 'q6LearnCodeAccelTrain',
       'q6LearnCodeDontKnowHowToYet', 'q6LearnCodeOther', 'q0006_other',
       'q8JobLevel', 'q0008_other', 'q8Student', 'q9CurrentRole',
       'q0009_other', 'q10Industry', 'q0010_other', 'q12JobCritPrefTechStack',
       'q12JobCritCompMission', 'q12JobCritCompCulture',
       'q12JobCritWorkLifeBal', 'q12JobCritCompensation',
       'q12JobCritProximity', 'q12JobCritPerks', 'q12JobCritSmartPeopleTeam',
       'q12JobCritImpactwithProduct', 'q12JobCritInterestProblems',
       'q12JobCritFundingandValuation', 'q12JobCritStability',
       'q12JobCritProfGrowth', 'q0012_other', 'q27EmergingTechSkill',
       'q0027_other', 'q28LoveC', 'q28LoveCPlusPlus', 'q28LoveJava',
       'q28LovePython', 'q28LoveRuby', 'q28LoveJavascript', 'q28LoveCSharp',
       'q28LoveGo', 'q28LoveScala', 'q28LovePerl', 'q28LoveSwift',
       'q28LovePascal', 'q28LoveClojure', 'q28LovePHP', 'q28LoveHaskell',
       'q28LoveLua', 'q28LoveR', 'q28LoveRust', 'q28LoveKotlin',
       'q28LoveTypescript', 'q28LoveErlang', 'q28LoveJulia', 'q28LoveOCaml',
       'q28LoveOther']
dataset = dataset[attributes]
## Checking the type for each attribute
dataset.info()
dataset[dataset['q4Education'] == 'Other (please specify)']
## Checking the 'q4Education' values:
dataset['q4Education'].unique()
## Dropping the 'q4Education' #NULL values:
ixNull = dataset[dataset['q4Education']=='#NULL!'].index
dataset = dataset.drop(labels=ixNull)
## Dropping the 'Other education level' column:
dataset = dataset.drop('q0004_other', axis=1)
dataset['q8JobLevel'].unique()
dataset['q0008_other'].unique()
## Counting the different employment levels:
q0008_total = dataset['q0008_other'].count()
q0008_unique = len(dataset['q0008_other'].unique())
print('From {0} different employment levels, {1} are unique.'.format(q0008_total, q0008_unique))
## Dropping down these instances:
q0008_indexes = dataset[dataset['q8JobLevel'] == dataset['q8JobLevel'].unique()[3]]['q0008_other'].index
dataset = dataset.drop(labels=q0008_indexes)
dataset = dataset.drop('q0008_other', axis=1)
indexq8JobLevel = dataset[dataset['q8JobLevel'] == 'Student'].index  #float64 type
indexq8Student = dataset[dataset['q8Student'] == 'Students'].index  #float64 type
np.unique(indexq8JobLevel == indexq8Student)
def clean_null(dataset):
    for col in dataset.columns:
        if '#NULL!' in dataset[col].unique():
            ixNull = dataset[dataset[col]=='#NULL!'].index
            dataset = dataset.drop(labels=ixNull)
            print('It was cleaned {0} null instances from {1}'.format(len(ixNull), col))
    return dataset
dataset = clean_null(dataset)
def map_q8JobLevel(ix):
    temp = hackerRank_numericMapping[(hackerRank_numericMapping['Data Field']=='q8JobLevel')&(hackerRank_numericMapping['Value']==ix)]['Label']
    temp = temp.values
    return temp[0]

def map_q4Education(ix):
    temp = hackerRank_numericMapping[(hackerRank_numericMapping['Data Field']=='q4Education')&(hackerRank_numericMapping['Value']==ix)]['Label']
    temp = temp.values
    return temp[0]
## Student - Under college:
education = [map_q4Education(1), map_q4Education(2)]
dataset.loc[(dataset['q8JobLevel']=='Student') & (dataset['q4Education'].isin(education)), 'Category'] = 'Student'
dataset.loc[(dataset['q8JobLevel']=='Student') & (dataset['q4Education'].isin(education)), 'Profile'] = 'Under college'
## Student - College:
education = [map_q4Education(3), map_q4Education(4), map_q4Education(5)]
dataset.loc[(dataset['q8JobLevel']=='Student') & (dataset['q4Education'].isin(education)), 'Category'] = 'Student'
dataset.loc[(dataset['q8JobLevel']=='Student') & (dataset['q4Education'].isin(education)), 'Profile'] = 'College'
## Student - Graduate:
education = [map_q4Education(6), map_q4Education(7)]
dataset.loc[(dataset['q8JobLevel']=='Student') & (dataset['q4Education'].isin(education)), 'Category'] = 'Student'
dataset.loc[(dataset['q8JobLevel']=='Student') & (dataset['q4Education'].isin(education)), 'Profile'] = 'Graduate'
## Professional - Junior:
joblevel = [map_q8JobLevel(2), map_q8JobLevel(4)]
dataset.loc[(dataset['q8JobLevel'].isin([2,4])), 'Category'] = 'Professional'
dataset.loc[(dataset['q8JobLevel'].isin([2,4])), 'Profile'] = 'Junior'
## Professional - Senior:
joblevel = [map_q8JobLevel(5), map_q8JobLevel(6), map_q8JobLevel(7), map_q8JobLevel(8)]
dataset.loc[(dataset['q8JobLevel'].isin(joblevel)), 'Category'] = 'Professional'
dataset.loc[(dataset['q8JobLevel'].isin(joblevel)), 'Profile'] = 'Senior'
## Professional - Freelancer:
dataset.loc[(dataset['q8JobLevel']==map_q8JobLevel(3)), 'Category'] = 'Professional'
dataset.loc[(dataset['q8JobLevel']==map_q8JobLevel(3)), 'Profile'] = 'Freelancer'
## Professional - Executive:
joblevel = [map_q8JobLevel(9), map_q8JobLevel(10)]
dataset.loc[(dataset['q8JobLevel'].isin(joblevel)), 'Category'] = 'Professional'
dataset.loc[(dataset['q8JobLevel'].isin(joblevel)), 'Profile'] = 'Executive'
def df_column_normalize(dataframe, percent=False):
    '''Normalizes the values of a given pandas.Dataframe by the total sum of each column.
    If percent=True, multiplies the final value by 100.
    Algorithm based on https://stackoverflow.com/questions/26537878/pandas-sum-across-columns-and-divide-each-cell-from-that-value'''
    if percent:
        return dataframe.div(dataframe.sum(axis=0), axis=1)*100
    else:
        return dataframe.div(dataframe.sum(axis=0), axis=1)
##Adjusting the data:
analysis01 = dataset[['q1AgeBeginCoding', 'q2Age','Category']]  #Copy the attributes of interest
analysis01['q1AgeBeginCoding'] = analysis01.q1AgeBeginCoding.apply(lambda x: x[:-9])  #Removing 'year old' text
analysis01['q2Age'] = analysis01.q2Age.apply(lambda x: x[:-9])  #Removing 'year old' text
## Checking the data before plotting:
analysis01.q2Age.unique()
## Drawing the barplots of ages for each class
fig1, (ax1,ax2) = plt.subplots(2,1,figsize=(12, 8))
fig1.subplots_adjust(top=.93)
plt.suptitle('Women age distribution among students and professionals', fontsize=14, fontweight='bold')

q2Age_order = ['12 - 18 ','18 - 24 ','25 - 34 ','35 - 44 ','45 - 54 ','55 - 64 ','75 years']
sns.countplot(x="q2Age", hue="Category", data=analysis01, ax=ax1, order=q2Age_order)
ax1.set_yticklabels(ax1.get_yticklabels(), ha="right", fontsize=12, weight='bold');
ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=12, weight='bold');
ax1.set_xlabel('Today Age', fontsize=13, weight='bold')

q1AgeBegin_order = ['5 - 10 ','11 - 15 ','16 - 20 ','21 - 25 ','26 - 30 ',
             '31 - 35 ','36 - 40 ','41 - 50 ','50+ years']
sns.countplot(x="q1AgeBeginCoding", hue="Category", data=analysis01, ax=ax2, order=q1AgeBegin_order)
ax2.set_yticklabels(ax2.get_yticklabels(), ha="right", fontsize=12, weight='bold');
ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=12, weight='bold');
ax2.set_xlabel('Age when started coding', fontsize=13, weight='bold')
analysis01_relation = analysis01.groupby(['q1AgeBeginCoding','q2Age']).Category.count()
analysis01_relation = analysis01_relation.unstack(1).replace(np.nan,0)
analysis01_relation = df_column_normalize(analysis01_relation, percent=True)
# Drawing a heatmap with the numeric values in each cell
fig2, ax = plt.subplots(figsize=(9,6))
fig2.subplots_adjust(top=.925)
plt.suptitle('From past to present: when women started coding and their ages', fontsize=14, fontweight='bold')

ax.set_yticklabels(ax.get_yticklabels(), ha="right", fontsize=12, weight='bold')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold')

cbar_kws = {'orientation':"vertical", 'pad':0.03, 'aspect':50}
sns.heatmap(analysis01_relation, annot=True, linewidths=.3, fmt='.2f', cmap='RdPu', ax=ax, cbar_kws=cbar_kws);

ax.set_ylabel('Age when started coding', fontsize=13, weight='bold')
ax.set_xlabel('Present Age', fontsize=13, weight='bold')
emergTech = dataset[['Category','q27EmergingTechSkill']]
analysis01 = emergTech.groupby('q27EmergingTechSkill').Category.value_counts()
analysis01 = analysis01.unstack()
analysis01.fillna(value=0, inplace=True)
analysis01 = df_column_normalize(analysis01, percent=True)
analysis01
# Drawing a heatmap with the numeric values in each cell
fig1, ax = plt.subplots(figsize=(4, 8))
fig1.subplots_adjust(top=.93)
plt.suptitle('Relative enrollment on emerging technologies by female students and professionals', fontsize=14, fontweight='bold')

ax.set_yticklabels(ax.get_yticklabels(), ha="right", fontsize=12, weight='bold')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold')

cbar_kws = {'orientation':"vertical", 'pad':0.1, 'aspect':50}
sns.heatmap(analysis01, annot=True, linewidths=.3, ax=ax, cmap='RdPu', cbar_kws=cbar_kws);
def df_row_normalize(dataframe):
    '''Normalizes the values of a given pandas.Dataframe by the total sum of each line.
    Algorithm based on https://stackoverflow.com/questions/26537878/pandas-sum-across-columns-and-divide-each-cell-from-that-value'''
    return dataframe.div(dataframe.sum(axis=1), axis=0)
language = dataset[['q28LoveC', 'q28LoveCPlusPlus',
       'q28LoveJava', 'q28LovePython', 'q28LoveRuby', 'q28LoveJavascript',
       'q28LoveCSharp', 'q28LoveGo', 'q28LoveScala', 'q28LovePerl',
       'q28LoveSwift', 'q28LovePascal', 'q28LoveClojure', 'q28LovePHP',
       'q28LoveHaskell', 'q28LoveLua', 'q28LoveR', 'q28LoveRust',
       'q28LoveKotlin', 'q28LoveTypescript', 'q28LoveErlang', 'q28LoveJulia',
       'q28LoveOCaml', 'q28LoveOther', 'Category']]
## Replacing all "hate" and "NaN" values by zero (we're interestede just in the languages they love, for while)
lovelanguage = language.replace('Hate',0)
lovelanguage = lovelanguage.replace('Love', 1)

## Replacing all "Love" and "NaN" values by zero (we're now interested just in the languages they hate)
hatelanguage = language.replace('Love',0)
hatelanguage = hatelanguage.replace('Hate', 1)
lovelanguage = lovelanguage.groupby('Category').sum()
lovelanguage = df_row_normalize(lovelanguage)*100
lovelanguage.reset_index(inplace=True)
hatelanguage = hatelanguage.groupby('Category').sum()
hatelanguage = df_row_normalize(hatelanguage)*100
hatelanguage.reset_index(inplace=True)
## Adjusting the columns names:
lovelanguage.columns
lovelanguage.columns = ['group','C', 'C++', 'Java', 'Python','Ruby', 'Javascript', 'C#', 'Go',
       'Scala', 'Perl', 'Swift', 'Pascal','Clojure', 'PHP', 'Haskell', 'Lua','R', 'Rust',
       'Kotlin', 'Typescript','Erlang', 'Julia', 'OCaml']
hatelanguage.columns = lovelanguage.columns
# From: https://python-graph-gallery.com/391-radar-chart-with-several-individuals/
categories=list(lovelanguage)[1:]
N = len(categories)
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
# Initialise the spider plot
fig3 = plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)
plt.title('Which programming language do women love the most?', fontsize=14, fontweight='bold')
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories) 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([5,10,15], ["5%","10%","15%"], color="grey", size=12)
plt.ylim(0,15)

# Plot each individual = each line of the data 
# Ind1
values=lovelanguage.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Professional")
ax.fill(angles, values, 'b', alpha=0.1)
# Ind2
values=lovelanguage.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Students")
ax.fill(angles, values, 'r', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
hatelanguage
# From: https://python-graph-gallery.com/391-radar-chart-with-several-individuals/
categories=list(hatelanguage)[1:]
N = len(categories)
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
# Initialise the spider plot
fig3 = plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)
plt.title('Which programming language do women hate the most?', fontsize=14, fontweight='bold')
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories) 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([3,6,9], ["3%","6%","9%"], color="grey", size=12)
plt.ylim(0,9)

# Plot each individual = each line of the data 
# Ind1
values=hatelanguage.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Professional")
ax.fill(angles, values, 'b', alpha=0.1)
# Ind2
values=hatelanguage.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Students")
ax.fill(angles, values, 'r', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))