#importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math

import warnings

warnings.filterwarnings('ignore')
# read the data

mcr =pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv',encoding='ISO-8859-1')

mcr.head()
mcr.info()
cols_to_drop = mcr[mcr.columns[pd.Series(mcr.columns).str.contains('OTHER_TEXT')]].columns.tolist()

mcr.drop(cols_to_drop,axis=1,inplace=True)
# function to plot barchart

from textwrap import wrap

def plot_bar(col,title,label='',rot=0):                        

    plt.figure(figsize=(16,6), dpi=80, facecolor='w')

    df1 = mcr[col][1:].value_counts()

    ax=round(100*df1/sum(df1),2).sort_values(ascending=False).plot.bar(stacked=True)

    labels = [item.get_text() for item in ax.get_xticklabels()]

    labels = ["\n".join(wrap(l,15)) for l in labels]

    plt.title(title)

    if label=='': label=mcr[col][0]

    plt.xlabel(label)

    plt.grid(b=True,which='major', color='#666666', linestyle='-',alpha=0.5)

    ax.set(ylabel='Percentage')

    ax.set_xticklabels(labels,rotation=rot)

    plt.minorticks_on()

    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    plt.tight_layout()

plot_bar('Q3','Country',label='Country',rot=90)
plot_bar('Q1','Age',label='Age groups')
plot_bar('Q4','Education')
plot_bar('Q2','Gender')
plot_bar('Q5','Current role')
plot_bar('Q6','Size of the company')
plot_bar('Q7','#Data Scientist')
plot_bar('Q8','Incorporate machine learning in business?')
plot_bar('Q10','Yearly Compensation',rot=60)
plot_bar('Q11','Money Spent on ML and CC platforms in last 5 years',label='Amount in $' )
plot_bar('Q15','Coding experience',label='# of Years' )
plot_bar('Q22','Used TPU?' )
plot_bar('Q23','Machine Learning experience',label='# of Years' )
def plot_bar_stacked(col,title,label='',rot=0):                        

                     

    plt.figure(figsize=(16,6), dpi=80, facecolor='w')

    #ax=sns.countplot(x=mcr[col][1:], data=mcr, order = mcr[col][1:].value_counts().index)

    df1 = mcr.groupby(col)['Q1'].value_counts()

    ax=round(100*df1/sum(df1),2).unstack().plot.bar(stacked=True)

    labels = [item.get_text() for item in ax.get_xticklabels()]

    labels = ["\n".join(wrap(l,15)) for l in labels]

    plt.title(title)

    if label=='': label=mcr[col][0]

    plt.xlabel(label)

    plt.grid(b=True,which='major', color='#666666', linestyle='-',alpha=0.5)

    ax.set(ylabel='Percentage')

    ax.set_xticklabels(labels,rotation=rot)

    plt.minorticks_on()

    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    plt.tight_layout()



      
India=mcr.loc[mcr['Q3']=='India']

USA=mcr.loc[mcr['Q3']=='United States of America']

India.head()
# Gender

plt.figure(figsize=(16,6), dpi=80, facecolor='w')

plt.subplot(121)

val = India['Q2'].value_counts()

label=list(India['Q2'].unique())

my_circle = plt.Circle((0, 0), 0.7, color = 'white')

plt.pie(val,autopct = '%.2f%%')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.legend(label,loc=2)

plt.title('Gender Distribution in India')

plt.subplot(122)

val = USA['Q2'].value_counts()

label=list(USA['Q2'].unique())

my_circle = plt.Circle((0, 0), 0.7, color = 'white')

plt.pie(val,autopct = '%.2f%%')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.legend(label,loc=2)

plt.title('Gender Distribution in USA')
# function to concat two dfs and plot

def func_plt_2df(col,tit,rot=0):

    fig, ax = plt.subplots(figsize=(16,10))

    ax= pd.concat({'India': 100*India[col].value_counts()/len(India[col]), 'USA': 100*USA[col].value_counts()/len(USA[col])}, axis=1).plot.bar(ax=ax)

    labels = [item.get_text() for item in ax.get_xticklabels()]

    labels = ["\n".join(wrap(l,15)) for l in labels]

    plt.grid(b=True,which='major', color='#666666', linestyle='-',alpha=0.5)

    plt.minorticks_on()

    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    ax.set(xlabel=mcr[col][0])

    ax.set_xticklabels(labels,rotation=rot)

    ax.set(ylabel='Percentage')

    plt.title(tit)
#Age groups

func_plt_2df('Q1','Age')
# Current Roles

func_plt_2df('Q5','Current role')
# Education

func_plt_2df('Q4','Education')
# Company Size 

func_plt_2df('Q6','Company Size ')
# individuals responsible for data science workloads

func_plt_2df('Q7','Number of individuals responsible for data science workloads')
# Incorporate ML metohods

func_plt_2df('Q8','Incorporate ML metohods')
func_plt_2df('Q10','Yearly Compensation',rot=90)

func_plt_2df('Q11','Money spent on Machine Learning and Cloud Computing platforms at work')
func_plt_2df('Q14','Primary tool use for Analysing Data')
func_plt_2df('Q15','Coding experience')
func_plt_2df('Q19','Programming Language recommended to aspiring Data Scientist')
func_plt_2df('Q22','TPU Usage')
func_plt_2df('Q23','Experience in using Machine Learning methods')
# func to plot questions with subdividsion

def func_grp_plt(column,tit):  

    cols = [col for col in mcr if col.startswith(column)]

    col=mcr[cols].iloc[0]

    col_key = [x.split('-')[2] for x in col]

        

    col_val=India[cols].count(axis=0)

    d=  {k:v for (k,v) in zip(col_key,col_val)}

    lists = sorted(d.items(),key=lambda x: x[1], reverse=True) # sorted by key, return a list of tuples

    x, y = zip(*lists)

    fig, ax = plt.subplots(figsize=(8,8))

    plt.barh(x, y)

    plt.xticks(rotation=0)   

    plt.xlabel('Frequency')

    plt.title(tit + ' - India')

    

    col_key = [x.split('-')[2] for x in col]

    col_val=USA[cols].count(axis=0)

    d=  {k:v for (k,v) in zip(col_key,col_val)}

    lists = sorted(d.items(),key=lambda x: x[1], reverse=True) # sorted by key, return a list of tuples

    x, y = zip(*lists)

    fig, ax = plt.subplots(figsize=(8,8))

    plt.barh(x, y)

    plt.xticks(rotation=0) 

    plt.xlabel('Frequency')

    plt.title(tit + ' - USA')





    
func_grp_plt('Q9','Imporatnt role at work')
func_grp_plt('Q13','Platforms in which data science courses were completed')
cols_Q14 = [col for col in mcr if col.startswith('Q14')]

col=mcr[cols_Q14].iloc[0]

col_key = [x.split('-')[1] for x in col]

col_val=India[cols_Q14].count(axis=0)

d=  {k:v for (k,v) in zip(col_key,col_val)}

lists = sorted(d.items(),key=lambda x: x[1], reverse=True) # sorted by key, return a list of tuples

x, y = zip(*lists)

fig, ax = plt.subplots(figsize=(8,8))

plt.barh(x, y)

col_val=USA[cols_Q14].count(axis=0)

d1=  {k:v for (k,v) in zip(col_key,col_val)}

lists = sorted(d.items(),key=lambda x: x[1], reverse=True) # sorted by key, return a list of tuples

x1, y1 = zip(*lists)

fig, ax = plt.subplots(figsize=(8,8))

#sns.barplot(x,y)

plt.barh(x, y)



plt.xticks(rotation=0)    



func_grp_plt('Q16','IDEs used regularly')
func_grp_plt('Q17','Hosted notebook products used regularly')
func_grp_plt('Q18','programming Languages used Regularly')
func_grp_plt('Q20','Data visulaisation libraries used regularly')
func_grp_plt('Q21','Specialised hardware used regularly')

func_grp_plt('Q24','ML algorithms used regularly')
func_grp_plt('Q25','ML tools used regularly')
func_grp_plt('Q26','Computer vision methods used regularly')
func_grp_plt('Q27','NLP methods used regularly')
func_grp_plt('Q28','Machine learning Frameworks used regularly')
func_grp_plt('Q29','Cloud computing platforms used regularly')
func_grp_plt('Q30','Specific cloud computing products used regularly')
func_grp_plt('Q31','Specific Big data/ analytics products used regularly')
func_grp_plt('Q32','Machine Learning Products used regularly')
func_grp_plt('Q33','Automated learning tools used regularly')
func_grp_plt('Q34','Relational database products used regularly')