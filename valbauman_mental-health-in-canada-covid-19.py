#analysis

import numpy as np 

import pandas as pd

import math

import scipy.stats as ss

from collections import Counter



#visualization

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input/camhdelvinia-survey-1'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
sur2= pd.read_csv('../input/camhdelvinia-survey-1/survey2_results.csv', encoding= 'unicode_escape')

sur2.columns
#distribution of gender identity

gender= sur2['Gender identity'].value_counts()

plt.pie(gender, labels= gender.index, explode= np.full((len(gender)), 0.1),labeldistance= None, radius=1.5)

plt.legend(gender.index, bbox_to_anchor=(0.8, 0.9))



#distribution of race

race= sur2['Q27 : Which of the following best describes your racial or ethnic group? '].value_counts()

plt.figure()

plt.pie(race, labels= race.index,radius=2)



#distribution of location

location= sur2['Province'].value_counts()

plt.figure()

plt.pie(location, labels= location.index, radius= 1.5)

print()



print(location)
def cond_entropy(x,y):

    #entropy of x given y

    #used to calculate the uncertainty coefficient

    y_count= Counter(y)

    xy_count= Counter(list(zip(x,y)))

    total_occurrences= sum(y_count.values())

    entropy= 0

    for xy in xy_count.keys():

        p_xy= xy_count[xy] / total_occurrences

        p_y= y_count[xy[1]] / total_occurrences

        entropy += p_xy*math.log(p_y/p_xy)

    return entropy



def uncertain_coeff(x,y):

    #returns number between 0 and 1.

    #0 means that feature y provides no info about feature x

    #1 means that feature y provides full info about feature x's value

    s_xy= cond_entropy(x,y)

    x_count= Counter(x)

    total_occurrences= sum(x_count.values())

    p_x= list(map(lambda n: n/total_occurrences, x_count.values()))

    s_x= ss.entropy(p_x)

    if s_x == 0:

        return 1

    else:

        return (s_x - s_xy) /s_x
q_cols= ['Q5 : How worried are you about the impact of COVID-19 on your personal financial situation?','Q6b : On average, how has the number of hours you are working for pay been affected by the COVID-19 pandemic? ','Q7 : How worried are you that you or someone close to you (close relative or friend) will get ill from COVID-19?','Q8x1 : P2W frequency - Feeling nervous, anxious or on edge','Q8x2 : P2W frequency - Not being able to stop or control worrying','Q8x3 : P2W frequency - Worrying too much about different things','Q8x4 : P2W frequency - Trouble relaxing','Q8x5 : P2W frequency - Being so restless that it?s hard to sit still','Q8x6 : P2W frequency - Becoming easily annoyed or irritable','Q8x7 : P2W frequency - Feeling afraid as if something awful might happen','Q20x1 : In the PAST 7 DAYS, how often have you felt depressed? ','Q20x2 : In the PAST 7 DAYS, how often have you felt lonely?','Q20x3 : In the PAST 7 DAYS, how often have you felt hopeful about the future? ','Q25 : What is the highest level of education you have completed?','Q26 : What is your current marital status?','Q27 : Which of the following best describes your racial or ethnic group? ']

corr= pd.DataFrame(index= q_cols, columns= q_cols)

for j in range(len(q_cols)):

    for k in range(len(q_cols)):

        u= uncertain_coeff(sur2[q_cols[j]].tolist(),sur2[q_cols[k]].tolist())

        corr.loc[q_cols[j],q_cols[k]]= u

corr.fillna(value= np.nan, inplace= True)

plt.figure(figsize=(10,10));sns.heatmap(corr,annot=True,fmt= '.2f')
#dataframes with data only from Quebec, Ontario, and Alberta

quebec= sur2.loc[(sur2['Province'] == 'Quebec')]

ontario= sur2.loc[(sur2['Province'] == 'Ontario')]

alberta= sur2.loc[(sur2['Province'] == 'Alberta')]



#function to return proportion of responses to a survey question

#(each survey question had multiple possible responses. this function returns the proportion of people that responded to each possible response)

def prop_responses(df, col):

    return df[col].value_counts()/np.shape(df)[0]



#function to plot proportion of responses

#first 4 arguments are dataframes of all survey responses for the 3 provinces of interest + Canada 

#q is column name (string) of the survey question to create plot for

#q_num is string of the question number

def plot_props(que, ont, alb, can, q, q_num):

    quebec= prop_responses(que, q).sort_index()

    ontario= prop_responses(ont, q).sort_index()

    alberta= prop_responses(alb, q).sort_index()

    all_prov= prop_responses(can, q).sort_index()

    

    bar_width= 0.2

    r1= np.arange(len(quebec))

    r2= [x + bar_width for x in r1]

    r3= [x + bar_width for x in r2]

    r4= [x + bar_width for x in r3]

    

    plt.figure()

    plt.bar(r1, quebec, width= bar_width, label= 'Quebec')

    plt.bar(r2,ontario,width=bar_width, label= 'Ontario')

    plt.bar(r3, alberta, width= bar_width, label= "Alberta")

    plt.bar(r4, all_prov, width= bar_width, label= 'All provinces')

    plt.xlabel('Response to Q'+ q_num); plt.ylabel('Proportion of responses')

    plt.legend()

    

    #correctly label the possible repsonses on the x-axis and give relevant title to plot

    if q_num in ['5','7']:

        plt.xticks([r+ bar_width for r in range(len(quebec))], ['Not at all','Not very','Somewhat','Very'])

        if q_num == '5':

            plt.title('Responses - how worried are you about the impact of COVID-19 on your personal finances')

        else:

             plt.title('Responses - how worried are you that you or someone close to you will get ill from COVID-19?')

    if q_num == '8':

        plt.xticks([r+ bar_width for r in range(len(quebec))], ['Nearly every day','Not at all','Over half','Several days'])

        plt.title('Responses - frequency of feeling nervous, anxious, or on edge')

    if q_num in ['20x1','20x3']:

        plt.xticks([r+ bar_width for r in range(len(quebec))], ['Most of the time','Occasionally','Rarely','Sometimes'])

        if q_num == '20x1':

            plt.title('Responses - how often have you felt depressed (past 7 days)')

        if q_num == '20x3':

            plt.title('Responses - how often have you felt hopeful about the future (past 7 days)')

    return



plot_props(quebec, ontario, alberta, sur2, 'Q5 : How worried are you about the impact of COVID-19 on your personal financial situation?', '5')

plot_props(quebec, ontario, alberta, sur2, 'Q7 : How worried are you that you or someone close to you (close relative or friend) will get ill from COVID-19?', '7')

plot_props(quebec, ontario, alberta, sur2, 'Q8x1 : P2W frequency - Feeling nervous, anxious or on edge', '8')

plot_props(quebec, ontario, alberta, sur2, 'Q20x1 : In the PAST 7 DAYS, how often have you felt depressed? ','20x1')

plot_props(quebec, ontario, alberta, sur2, 'Q20x3 : In the PAST 7 DAYS, how often have you felt hopeful about the future? ', '20x3')
#function to calculate confidence interval for a difference in proportions between two groups

#if the proportions of the two groups are different, prints a statement saying this and specifying how the two groups are different

#if 0 is in the interval, there is strong evidence to suggest that the two proportions are the same, meaning both groups being considered should be prioritized equally

#arguments: group1/group 2 are strings specifying the name of the groups under consideration. resp1/resp2 are the numbers of people that gave the worst response in the two groups. size1/size2 are the total size of the two groups

#arguments cont: q is survey question under consideration (string), alpha is level of significance

def two_prop_CI(group1, resp1, size1, group2, resp2, size2, q, alpha= 0.05):

    prop1= resp1/size1

    prop2= resp2/size2

    var= prop1 * (1-prop1) / size1 + prop2 * (1-prop2) / size2

    stdev= var**0.5

    

    z= ss.norm(loc=0, scale= 1).ppf((1-alpha)+(alpha/2))

    

    diff= prop2 - prop1

    CI= diff + np.array([-1,1])*z*stdev

    

    if CI[0] < 0:

        if CI[1] >0:

            #print('No difference in proportion of responses to '+ q + '. Applies to '+ group1 + ' and '+ group2+ '. Equally prioritize mental health initiatives in these areas')

            return

        elif CI[1] <0:

            print('Greater proportion of negative responses from '+ group2+' compared to '+ group1+' for '+q)

            return

    elif CI[0] > 0:

        print('Greater proportion of negative responses from '+ group1+' compared to '+ group2+' for '+q)

        return

    return 



#number of participants from the three provinces and total number of participants

que_size= np.shape(quebec)[0]

ont_size= np.shape(ontario)[0]

alb_size= np.shape(alberta)[0]

can_size= np.shape(sur2)[0]



#5 survey questions being considered

q5= 'Q5 : How worried are you about the impact of COVID-19 on your personal financial situation?'

q7= 'Q7 : How worried are you that you or someone close to you (close relative or friend) will get ill from COVID-19?'

q8= 'Q8x1 : P2W frequency - Feeling nervous, anxious or on edge'

q20x1= 'Q20x1 : In the PAST 7 DAYS, how often have you felt depressed? '

q20x3= 'Q20x3 : In the PAST 7 DAYS, how often have you felt hopeful about the future? '



for i in [q5, q7, q8, q20x1, q20x3]:

    if i in [q5, q7]:

        idx= -1 #idx is the index of the "worst" response in value_counts()

    elif i in [q8, q20x1]:

        idx= 0

    else: #only 20x3 remains

        idx= 2

    

    #number of responses to the most negative response to the survey question

    que_resp= quebec[i].value_counts().sort_index()[idx]

    ont_resp= ontario[i].value_counts().sort_index()[idx]

    alb_resp= alberta[i].value_counts().sort_index()[idx]

    can_resp= sur2[i].value_counts().sort_index()[idx]

    

    two_prop_CI('Quebec',que_resp,que_size,'Ontario',ont_resp,ont_size,i)

    two_prop_CI('Quebec', que_resp, que_size, 'Alberta', alb_resp, alb_size, i)

    two_prop_CI('Quebec', que_resp, que_size, 'across Canada', can_resp, can_size, i)

    two_prop_CI('Alberta',alb_resp,alb_size,'Ontario',ont_resp,ont_size,i)

    two_prop_CI('Alberta',alb_resp,alb_size,'across Canada',can_resp,can_size,i)

    two_prop_CI('Ontario',ont_resp,ont_size,'across Canada',can_resp,can_size,i)

kids= sur2.loc[(sur2['hChildren : Do you have children living in your household?'] == 'Kids')]

nokids= sur2.loc[(sur2['hChildren : Do you have children living in your household?'] == 'No Kids')]



k_size= np.shape(kids)[0]

nk_size= np.shape(nokids)[0]



for i in [q5, q7, q8, q20x1, q20x3]:

    if i in [q5, q7]:

        idx= -1 #idx is the index of the "worst" response in value_counts()

    elif i in [q8, q20x1]:

        idx= 0

    else: #only 20x3 remains

        idx= 2

    

    #number of responses to the most negative response to the survey question

    k_resp= kids[i].value_counts().sort_index()[idx]

    nk_resp= nokids[i].value_counts().sort_index()[idx]

    

    two_prop_CI('Kids', k_resp, k_size, 'No kids', nk_resp, nk_size, i)