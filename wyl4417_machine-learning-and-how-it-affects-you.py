# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import cufflinks as cf

import seaborn.apionly as sns

import scipy as sp

import matplotlib.patches as mpatches

%matplotlib inline
import pandas as pd

multi = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

multi_2= pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

multi_3= pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")
#multiple ans question

mulans= {'Q9' : range(11,19),'Q12' : range(22,34),'Q13' :range(35,47),'Q16':range(56,68),'Q17':range(69,81),'Q18':range(82,94),'Q20':range(97,109),'Q21':range(110,115),'Q24':range(118,130),'Q25':range(131,139),'Q26':range(140,147),'Q27':range(148,154),'Q28':range(155,167),'Q29':range(168,180),'Q30':range(181,193),'Q31':range(194,206),'Q32':range(207,219),'Q33':range(220,232),'Q34':range(233,245)}

#one answer question

oneans= list(multi.columns[1:11])+list(multi.columns[20:22])+list(multi.columns[48:49])+list(multi.columns[55:56])+list(multi.columns[95:96])+list(multi.columns[116:117])+list(multi.columns[117:118])



multi.rename(columns ={'Time from Start to Finish (seconds)':'number'},inplace = True)

multi_2.rename(columns ={'Time from Start to Finish (seconds)':'number'},inplace = True)

multi_3.rename(columns ={'Time from Start to Finish (seconds)':'number'},inplace = True)

multi.columns
def convert(s): 

    str1 = "" 

    return(str1.join(s)) 



def item_name(x):

    languages = []

    for col in multi.iloc[0,mulans[x]]:

        lal = col.split("-")

        doom = lal[2]

        doom_2 = list(doom)

        del doom_2[0]

        doom_3 = convert(doom_2)

        languages.append(doom_3)

    return languages



def perfect(j):

    nice = []

    for col in list(multi.columns[mulans[j]]):

        for col2 in item_name(j):

            q_1 =multi.loc[multi[col] == col2]

            q_1.rename(columns={col: j},inplace = True)

            nice.append(q_1)

    nice = pd.concat(nice)

    return nice

duit = perfect('Q9')
q9_2_d = multi.loc[multi['Q9_Part_2'] != 'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data']

q9_3_d = q9_2_d.loc[q9_2_d['Q9_Part_3'] != 'Build prototypes to explore applying machine learning to new areas']

q9_4_d = q9_3_d.loc[q9_3_d['Q9_Part_4'] != 'Build and/or run a machine learning service that operationally improves my product or workflows']

q9_5_d = q9_4_d.loc[q9_4_d['Q9_Part_5'] != 'Experimentation and iteration to improve existing ML models']

q9_6_d = q9_5_d.loc[q9_5_d['Q9_Part_6'] != 'Do research that advances the state of the art of machine learning']

q9_7_d = q9_6_d.loc[q9_6_d['Q9_Part_7'] != 'None of these activities are an important part of my role at work']

q9_8_d = q9_7_d.loc[q9_7_d['Q9_Part_8'] != 'Other']

q9_9_d = q9_8_d.loc[q9_7_d['Q9_Part_1'] == 'Analyze and understand data to influence product or business decisions']



we = pd.pivot_table(q9_9_d,values=['number'], index=['Q8'],aggfunc= lambda x :len(x))

we_t = pd.pivot_table(duit,values=['number'], index=['Q8','Q9'],aggfunc= lambda x :len(x))





we_r = we_t.reset_index()

we_a = we_r.loc[we_r['Q9'] != 'Other']

we_b = we_a.set_index('Q8')

we_group = we_b.groupby('Q8').sum()



data = we['number']/we_group['number']

data_1 = data.reset_index()

data_2 = data_1.sort_values('number', ascending = False)

plt.figure(figsize =(10,7))

axf = sns.barplot(x='Q8',y = 'number', data = data_2, palette = 'Spectral')

x_axis=range(4)

labels = data_2['Q8']

import textwrap

from  textwrap import fill

plt.xticks(x_axis, [textwrap.fill(label, 20) for label in labels], 

           rotation = 0, fontsize=12, horizontalalignment="center")

plt.title('Percentage of workers who analyse data only, for different companies with different levels of machine learning involvement', size = 10)

plt.ylabel(' % Number of Respondents',size = 10)

plt.xlabel('Figure 1',size = 10)
multi['Q4'].replace({'Bachelor’s degree':'University degree'

                              ,'Doctoral degree':'University degree'

                              ,'Master’s degree':'University degree'

                              ,'Professional degree':'University degree'

                              ,'I prefer not to answer':'Without University degree'

                              ,'No formal education past high school':'Without University degree'

                             , 'Some college/university study without earning a bachelor’s degree':'Without University degree'},inplace = True)

plp = pd.pivot_table(multi[1:],values=['number'], index=['Q4','Q10'],aggfunc= lambda x :len(x))

plp_1 = plp.unstack()

plp_1.columns = plp_1.columns.droplevel()

plp_2 = plp_1.transpose()

plp_2 = plp_2.div(plp_2.sum())*100

xaxis1 = ["0 -1K", "1 - 2K", "2 - 3K", "3 - 4K", "4 - 5K", "5 - 7.5K", "7.5 - 10K", "10 - 15K", "15 - 20K", "20 - 25K", "25 - 30K", "30 - 40K", "40 - 50K", "50 - 60K", "60 - 70K", "70 - 80K", "80 - 90K", "90 - 100K", "100 - 125K", "125 - 150K", "150 - 200K", "200 - 250K", "250 - 300K", "300 - 500K", "> $500K"]

plp_3 = plp_2.rename(index = {'$0-999':'0-999','> $500,000':'greater than 500,000'})

plp_4 = plp_3.reset_index()

plp_4['length'] = plp_4['Q10'].str.len()

plp_5 = plp_4.sort_values(['length','Q10'])

plp_5['money'] = xaxis1

plp_a = plp_5.set_index('money')

plp_b = plp_a.drop(['Q10'],axis = 1)

plp_6= plp_b.drop(columns=['length'])

axa = plp_6.plot(kind ='bar',stacked = False,figsize = (13,7),cmap = "Set3",alpha = 0.85, width = 0.85)

axa.legend(loc = 'center left',bbox_to_anchor =(1, 0.5))

axa.set_xticklabels(axa.get_xticklabels(), rotation=45)

plt.title('Compensation of respondents who graduated with / without a university degree ', size = 15)

plt.ylabel('% Number of Respondents',size = 13)

plt.xlabel('Figure 2',size = 10)
indx = 'Q5' #Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?

def convert(s): 

    str1 = "" 

    return(str1.join(s)) 



if indx in oneans:

    bob = pd.pivot_table(multi[1:],values=['number'], index=indx,aggfunc= lambda x :len(x))

else:

    win=[]

    for col in multi.iloc[0,mulans[indx]]:

        lal = col.split("-")

        doom = lal[2]

        doom_2 = list(doom)

        del doom_2[0]

        doom_3 = convert(doom_2)

        win.append(doom_3)

    

    result = []

    for col in mulans[indx]:

        we =[]

        for row in range(1,19718):

            lal = multi.iloc[row,col]

            we.append(lal)

        withoutna = [x for x in we if str(x) != 'nan']

        result.append(len(withoutna))

    bob=pd.DataFrame(data=result,index=win) 

bob_1 = bob.reset_index()

bob_2 = bob_1.sort_values('number')

plt.figure(figsize = (13,8))

axs = sns.barplot(x='Q5',y = 'number', data = bob_2, palette = 'Spectral')

for p in axs.patches:

    axs.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'top', xytext = (0, 10), textcoords = 'offset points')

plt.title('Jobs of respondents of the survey',size = 15)

plt.ylabel('Number of respondents',size = 13)

x_axis = range(12)

labels = bob_2['Q5']

import textwrap

from  textwrap import fill

plt.xticks(x_axis, [textwrap.fill(label, 13) for label in labels], 

           rotation = 0, fontsize=12, horizontalalignment="center")

axs.set_xticklabels(axs.get_xticklabels(), rotation = 50)

plt.xlabel('Figure 3',size = 10)
student_compare = pd.pivot_table(multi_2[1:],values=['number'], index=['Q5','Q15'],aggfunc= lambda x :len(x))

sc = student_compare.loc['Student']

sc_1 = sc.loc['I have never written code']

sc_2 = sc.iloc[0:6]

sc_3 = sc_2.sum()

sc_4 = pd.concat([sc_1,sc_3],axis =1)

sc_4.rename(columns={0: "Students who have written code",'I have never written code':'Students who never written code'},inplace = True)

sc_4.rename(index={'number': "student"},inplace = True)

sc_5 = sc_4.transpose()

pie_label = sc_5['student'].sort_values().index

pie_counts = sc_5['student'].sort_values()

cmap = plt.get_cmap('Set3')

colors = [cmap(i) for i in np.linspace(0, 1, 8)]

student_pit = plt.pie(pie_counts, labels = pie_label, autopct='%1.1f%%', shadow=True, colors=colors)

plt.title('Percentage of students who have written code / never written code before',size = 13)

plt.xlabel('Figure 4',size = 10)
duit = perfect('Q9')

business_duit =  pd.pivot_table(duit,values=['number'], index=['Q5','Q9'],aggfunc= lambda x :len(x))



st = business_duit.loc['Business Analyst']

st_1 = st.reset_index()

plt.figure(figsize = (13,8))

axy = sns.barplot(x='Q9',y = 'number', data = st_1, palette = 'Spectral')

for p in axy.patches:

    axy.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'top', xytext = (0, 10), textcoords = 'offset points')

plt.title('Job scope of respondents who are business analysts' , size = 15)

plt.xlabel('Figure 5',size = 13)

plt.ylabel('Number of respondents',size = 13)

x_axis = range(8)

labels = st_1['Q9']

import textwrap

from  textwrap import fill

plt.xticks(x_axis, [textwrap.fill(label, 13) for label in labels], 

           rotation = 0, fontsize=12, horizontalalignment="center")



plt.xlabel('Figure 5',size = 10)

research = pd.pivot_table(multi_3[1:],values=['number'], index=['Q9_Part_6','Q4'],aggfunc= lambda x :len(x))

research_1 = research.reset_index()

research_2 = research_1.sort_values('number')

plt.figure(figsize = (12,8))

re = sns.barplot(x='Q4',y = 'number', data = research_2, palette = 'Spectral')

for p in re.patches:

    re.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

labels = research_2['Q4']

x_axis=range(8)

import textwrap

from  textwrap import fill

plt.xticks(x_axis, [textwrap.fill(label, 13) for label in labels], 

           rotation = 0, fontsize=12, horizontalalignment="center")

plt.title('Education level of respondents who does research in state of art machine learning' , size = 15)

plt.ylabel('Number of respondents',size = 13)

plt.xlabel('Figure 6',size = 10)
toom = pd.pivot_table(multi[1:],values=['number'], index=['Q23','Q10'],aggfunc= lambda x :len(x))

toom_1 = toom.unstack()

toom_1.columns = toom_1.columns.droplevel()

toom_1 = toom_1.reindex(index = ['< 1 years',

 '1-2 years',

'2-3 years',

 '3-4 years',

'4-5 years',

 '5-10 years',

 '10-15 years',

 '20+ years'])



toom_1 = toom_1[['$0-999',

 '1,000-1,999',

 '2,000-2,999',

 '3,000-3,999',

 '4,000-4,999',

 '5,000-7,499',

 '7,500-9,999',

 '10,000-14,999',

 '15,000-19,999',

 '20,000-24,999',

 '25,000-29,999',

 '30,000-39,999',

 '40,000-49,999',

 '50,000-59,999',

 '60,000-69,999',

 '70,000-79,999',

 '80,000-89,999',

 '90,000-99,999',

 '100,000-124,999',

 '125,000-149,999',

 '150,000-199,999',

 '200,000-249,999',

 '250,000-299,999',

 '300,000-500,000','> $500,000']]



toom_1 = toom_1.div(toom_1.sum())

toom_2 = toom_1.transpose()

toom_3 = toom_2.reset_index()

xaxis1 = ["0 -1K", "1 - 2K", "2 - 3K", "3 - 4K", "4 - 5K", "5 - 7.5K", "7.5 - 10K", "10 - 15K", "15 - 20K", "20 - 25K", "25 - 30K", "30 - 40K", "40 - 50K", "50 - 60K", "60 - 70K", "70 - 80K", "80 - 90K", "90 - 100K", "100 - 125K", "125 - 150K", "150 - 200K", "200 - 250K", "250 - 300K", "300 - 500K", "> $500K"]

toom_3['money'] = xaxis1

toom_4 = toom_3.set_index('money')

toom_4 = toom_4.drop(['Q10'],axis = 1)



axm = toom_4.plot(kind ='bar',stacked =True,figsize = (13,7),cmap = "Set3",alpha = 0.85, width = 0.85)

axm.legend(loc = 'center left',bbox_to_anchor =(1, 0.5))

axm.set_xticklabels(axm.get_xticklabels(), rotation=45)

plt.title('Correlation between machine learning experience, number of respondents and yearly income', size = 15)

plt.ylabel(' % Number of Respondents',size = 13)

plt.xlabel('Figure 7',size = 10)
boom = pd.pivot_table(multi[1:],values=['number'], index=['Q10','Q15'],aggfunc= lambda x :len(x))

boom_1 = boom.unstack()

boom_1.columns = boom_1.columns.droplevel()

boom_2 = boom_1.rename(index = {'$0-999':'0-999','> $500,000':'greater than 500,000'})

boom_3 = boom_2.reset_index()

xaxis1 = ["0 -1K", "1 - 2K", "2 - 3K", "3 - 4K", "4 - 5K", "5 - 7.5K", "7.5 - 10K", "10 - 15K", "15 - 20K", "20 - 25K", "25 - 30K", "30 - 40K", "40 - 50K", "50 - 60K", "60 - 70K", "70 - 80K", "80 - 90K", "90 - 100K", "100 - 125K", "125 - 150K", "150 - 200K", "200 - 250K", "250 - 300K", "300 - 500K", "> $500K"]

boom_3['length'] = boom_3['Q10'].str.len()

boom_4 = boom_3.sort_values(['length','Q10'])

boom_5= boom_4.drop(columns=['length'])

boom_5['money'] = xaxis1

boom_6 = boom_5.set_index('money')

boom_6 = boom_6[['I have never written code', '< 1 years','1-2 years','3-5 years','5-10 years','10-20 years','20+ years']]

boom_7 = boom_6.transpose()

boom_7 = boom_7.div(boom_7.sum())

toom_3['money'] = xaxis1

boom_8=boom_7.transpose()

axn = boom_8.plot(kind ='bar',stacked =True,figsize = (13,7),cmap = "Set3",alpha = 0.85, width = 0.85)

axn.legend(loc = 'center left',bbox_to_anchor =(1, 0.5))

axn.set_xticklabels(axn.get_xticklabels(), rotation=45)

plt.title('Correlation between coding experience, number of respondents and yearly income', size = 15)

plt.ylabel(' % Number of Respondents',size = 13)

plt.xlabel('Figure 8',size = 10)
q9_2_d = multi.loc[multi['Q9_Part_2'] != 'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data']

q9_3_d = q9_2_d.loc[q9_2_d['Q9_Part_3'] != 'Build prototypes to explore applying machine learning to new areas']

q9_4_d = q9_3_d.loc[q9_3_d['Q9_Part_4'] != 'Build and/or run a machine learning service that operationally improves my product or workflows']

q9_5_d = q9_4_d.loc[q9_4_d['Q9_Part_5'] != 'Experimentation and iteration to improve existing ML models']

q9_6_d = q9_5_d.loc[q9_5_d['Q9_Part_6'] != 'Do research that advances the state of the art of machine learning']

q9_7_d = q9_6_d.loc[q9_6_d['Q9_Part_7'] != 'None of these activities are an important part of my role at work']

q9_8_d = q9_7_d.loc[q9_7_d['Q9_Part_8'] != 'Other']

q9_9_d = q9_8_d.loc[q9_7_d['Q9_Part_1'] == 'Analyze and understand data to influence product or business decisions']



year = pd.pivot_table(q9_9_d[1:],values=['number'], index=['Q10'],aggfunc= lambda x :len(x))

year.rename(columns={"number": "analyze data only"},inplace = True)

year_1 = year.rename(index = {'$0-999':'0-999','> $500,000':'greater than 500,000'})

year_1.reset_index(inplace= True)

year_1['length'] = year_1['Q10'].str.len()

year_2 = year_1.sort_values(['length','Q10'])

year_3= year_2.drop(columns=['length'])

year_3.set_index('Q10',inplace=True)



year_tot = pd.pivot_table(multi[1:],values=['number'], index=['Q10'],aggfunc= lambda x :len(x))

year_tot.rename(columns={"number": "all"},inplace = True)



year_tot_1 = year_tot.rename(index = {'$0-999':'0-999','> $500,000':'greater than 500,000'})

year_tot_1.reset_index(inplace= True)

year_tot_1['length'] = year_tot_1['Q10'].str.len()

year_tot_2 = year_tot_1.sort_values(['length','Q10'])

year_tot_3= year_tot_2.drop(columns=['length'])

year_tot_3.set_index('Q10',inplace=True)



tgt = pd.concat([year_tot_3,year_3],axis = 1)

tgt_1 = tgt.reset_index()

tgt_1['total'] = tgt_1['all'] + tgt_1['analyze data only']

tgt_1['% all'] = tgt_1['all']/tgt_1['total']

tgt_1['% analyze data only'] = tgt_1['analyze data only']/tgt_1['total']

tgt_2 = tgt_1.drop(columns=['all', 'analyze data only','total'])

tgt_3 = tgt_2.set_index('Q10')

ax4 = tgt_3.plot(kind ='bar',stacked = True,figsize = (13,7),rot = 1,cmap = "Set3",alpha = 0.85, width = 0.85,label = xaxis1)

ax4.legend(loc = 'center left',bbox_to_anchor =(1, 0.5)) 

labels = research_2['Q4']

x_axis=range(8)

ax4.set_xticklabels(axn.get_xticklabels(), rotation=45)

plt.xlabel('Figure 9',size = 10)

indx = 'Q15' #Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?

def convert(s): 

    str1 = "" 

    return(str1.join(s)) 



if indx in oneans:

    tob = pd.pivot_table(multi_3[1:],values=['number'], index=indx,aggfunc= lambda x :len(x))

else:

    win=[]

    for col in multi_3.iloc[0,mulans[indx]]:

        lal = col.split("-")

        doom = lal[2]

        doom_2 = list(doom)

        del doom_2[0]

        doom_3 = convert(doom_2)

        win.append(doom_3)

    

    result = []

    for col in mulans[indx]:

        we =[]

        for row in range(1,19718):

            lal = multi_3.iloc[row,col]

            we.append(lal)

        withoutna = [x for x in we if str(x) != 'nan']

        result.append(len(withoutna))

    tob=pd.DataFrame(data=result,index=win) 



tob_1= tob.reindex (index = ['I have never written code', '< 1 years','1-2 years','3-5 years','5-10 years','10-20 years','20+ years'])

tob_2 = tob_1.reset_index()

plt.figure(figsize = (12,8))

axc = sns.barplot(x='Q15',y = 'number', data = tob_2, palette = 'Spectral')

for p in axc.patches:

    axc.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    

plt.title('Coding experience', size = 15)

plt.ylabel('Number of respondents',size = 13)

plt.xlabel('Figure 10',size = 10)

w = perfect('Q9')

count_q9 = pd.pivot_table(w[1:],values=['number'], index=['Q3','Q9'],aggfunc= lambda x :len(x))

q9_india = count_q9.iloc[150:158]

q9_india_1 = q9_india.unstack(level = -1)

q9_usa = count_q9.iloc[435:443]

q9_usa_1 = q9_usa.unstack(level = -1)

q9_tpt = pd.concat([q9_india_1,q9_usa_1])

q9_tpt.columns = q9_tpt.columns.droplevel()

q9_tot = q9_tpt.transpose()

q9_tot = q9_tot.div(q9_tot.sum())*100

q9_tot_1 = q9_tot.reset_index()

plt.figure(figsize = (13,8))

axr = q9_tot_1.plot(kind ='bar',stacked = False,figsize = (13,7),cmap = "Set3",alpha = 0.85, width = 0.70)

axr.legend(loc = 'center left',bbox_to_anchor =(1, 0.5))

labels = q9_tot.index

x_axis=range(8)

import textwrap

from  textwrap import fill

plt.xticks(x_axis, [textwrap.fill(label, 13) for label in labels], 

           rotation = 0, fontsize=12, horizontalalignment="center")

for p in axr.patches:

    axr.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('India and USA', size = 15)

plt.ylabel(' % Number of Respondents',size = 13)

plt.xlabel('Figure 11',size = 10)
count_q5 = pd.pivot_table(multi[1:],values=['number'], index=['Q3','Q5'],aggfunc= lambda x :len(x))



q5_india = count_q5.iloc[222:234]

q5_usa = count_q5.iloc[635:647]

q5_india_1 = q5_india.unstack(level = -1)

q5_usa_1 = q5_usa.unstack(level = -1)

q5_tpt = pd.concat([q5_india_1,q5_usa_1])

q5_tpt.columns = q5_tpt.columns.droplevel()

q5_tot = q5_tpt.transpose()

q5_tot = q5_tot.div(q5_tot.sum())*100

q5_tot_1 = q5_tot.reset_index()

plt.figure(figsize = (13,8))

axr = q5_tot_1.plot(kind ='bar',stacked = False,figsize = (13,7),cmap = "Set3",alpha = 0.85, width = 0.70)

axr.legend(loc = 'center left',bbox_to_anchor =(1, 0.5))

labels = q5_tot.index

x_axis=range(12)

import textwrap

from  textwrap import fill

plt.xticks(x_axis, [textwrap.fill(label, 13) for label in labels], 

           rotation = 0, fontsize=12, horizontalalignment="center")

for p in axr.patches:

    axr.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('India and USA', size = 15)

plt.ylabel(' % Number of Respondents',size = 13)

axr.set_xticklabels(axr.get_xticklabels(), rotation=45)

plt.xlabel('Figure 12',size = 10)
