# Disable warnings in Anaconda

#import warnings

#warnings.filterwarnings('ignore')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Graphics in retina format are more sharp and legible

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

plt.rcParams['image.cmap'] = 'viridis'





import plotly.offline as py

import pycountry



py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from plotly.offline import init_notebook_mode, iplot 

init_notebook_mode(connected=True)



import folium 

from folium import plugins
#Importing the 2019 Dataset

df_2019 =  pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

df_2019.columns = df_2019.iloc[0]

df_2019=df_2019.drop([0])



#df_2018 = pd.read_csv('/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv')

#df_2018.columns = df_2018.iloc[0]

#df_2018=df_2018.drop([0])



#df_2017=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')
print('Total respondents in 2019:',df_2019.shape[0])
df_2019['In which country do you currently reside?'].replace({'United States of America':'United States','Viet Nam':'Vietnam','China':"People 's Republic of China","United Kingdom of Great Britain and Northern Ireland":'United Kingdom',"Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)



topn = 10

male = df_2019[df_2019['What is your gender? - Selected Choice']=='Male']

female = df_2019[df_2019['What is your gender? - Selected Choice']=='Female']

count_male = male['In which country do you currently reside?'].value_counts()[:topn].reset_index()

count_female = female['In which country do you currently reside?'].value_counts()[:topn].reset_index()



pie_men = go.Pie(labels=count_male['index'],values=count_male['In which country do you currently reside?'],name="Men",hole=0.5,domain={'x': [0,0.46]})

pie_women = go.Pie(labels=count_female['index'],values=count_female['In which country do you currently reside?'],name="Women",hole=0.5,domain={'x': [0.52,1]})



layout = dict(title = 'Top-10 countries with respondents', font=dict(size=12), legend=dict(orientation="h"),

              annotations = [dict(x=0.2, y=0.5, text='Men', showarrow=False, font=dict(size=20)),

                             dict(x=0.8, y=0.5, text='Women', showarrow=False, font=dict(size=20)) ])



fig = dict(data=[pie_men, pie_women], layout=layout)

py.iplot(fig)
df_2019.head()
colors = ['#1BA1E2', '#AA00FF', '#F0A30A','#8c564b', '#622c70', '#B83B5E', '#F08A5D','#F372AP','#D597CE','#F38181','#FCE38A','#8EF6E4','#F67280'] #gold,bronze,silver,chestnut brown

counts = df_2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts(sort=True)

labels = counts.index

values = counts.values



pie = go.Pie(labels=labels, values=values, marker=dict(colors=colors))

layout = go.Layout(title='Role Distribution in 2019')

fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)
def return_percentage(data,question_part,response_count):

    """Calculates percent of each value in a given column"""

    counts = data[question_part].value_counts()

    total = response_count

    percentage = (counts*100)/total

    value = [percentage]

    question = [data[question_part]][0]

    percentage_df = pd.DataFrame(data=value).T     

    return percentage_df





def plot_multiple_choice(data,question,title,y_axis_title):

    df = return_percentage(data,question,response_count)

    

    trace1 = go.Bar(

                    x = df.index,

                    y = df[question][0:10],

                    name = "Kaggle Survey 2019",

                    marker = dict(color='#AA00FF',

                                 line=dict(color='black',width=1.5)),

                    text = df.index)

    data = [trace1]

    layout = go.Layout(barmode = "group",title=title, 

                       yaxis= dict(title=y_axis_title),showlegend=False)

    fig = go.Figure(data = data, layout = layout)

    iplot(fig)



def multivariate_plot(label,title,xlabel,ylabel):

                                

    ax = label.plot(kind='bar',width=0.8)

    plt.gcf().set_size_inches(16,8)

    plt.xticks( rotation=45,fontsize='10', horizontalalignment='right')

    plt.yticks( fontsize=10)

    plt.title(title, fontsize = 15)

    plt.xlabel(xlabel,fontsize=15)

    plt.ylabel(ylabel,fontsize=15)

    plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="centre left")

    plt.show()

    
df_edu_temp = pd.crosstab(df_2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'],

              df_2019['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'])





df_edu = df_edu_temp[(df_edu_temp.index == 'Business Analyst')| (df_edu_temp.index == 'DBA/Database Engineer') | (df_edu_temp.index == 'Data Analyst') | (df_edu_temp.index == 'Data Engineer') | (df_edu_temp.index == 'Data Scientist')].drop('I prefer not to answer',axis=1)

                    







ax = df_edu.plot(kind='bar',width=1)

plt.gcf().set_size_inches(16,8)

plt.xticks( rotation=45,fontsize='10', horizontalalignment='right')

plt.yticks( fontsize=10)

plt.title('Role wise Education Distribution', fontsize = 15)

plt.xlabel('Roles',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="center left")

plt.show()
#money spent of the Machine learning vs Education and roles 

df_edu_temp = pd.crosstab(df_2019['What is your current yearly compensation (approximate $USD)?'],

              df_2019['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'])





df_edu = df_edu_temp[(df_edu_temp.index == '$0-999')| (df_edu_temp.index == '1,000-1,999') | (df_edu_temp.index == '10,000-14,999') | (df_edu_temp.index == '15,000-19,999')|(df_edu_temp.index == '125,000-149,999')|(df_edu_temp.index == '<500000')|(df_edu_temp.index == '150,000-199,999')|(df_edu_temp.index == '2,000-2,999')

        | (df_edu_temp.index == '20,000-24,999') | (df_edu_temp.index == '25,000-29,999')| (df_edu_temp.index == '200,000-249,999')  

        | (df_edu_temp.index == '3,000-3,999') | (df_edu_temp.index == '30,000-34,999') | (df_edu_temp.index == '300,000-500,000')|(df_edu_temp.index == '125,000-149,999')| (df_edu_temp.index == '4,000-4,999')| (df_edu_temp.index == '40,000-49,999')]

                    



#df_edu_temp = df_edu.sort_values('What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',ascending=False)

ax = df_edu.plot(kind='bar',width=0.4, align= 'center')

plt.gcf().set_size_inches(36,20)

plt.xticks( rotation=0,fontsize='20', horizontalalignment='right')

plt.yticks( fontsize=10)



plt.title('Role wise Education Distribution', fontsize = 25)

plt.xlabel('Roles',fontsize=35)

plt.ylabel('Count',fontsize=35)

plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="centre left")

plt.show()
df_sal_temp = pd.crosstab(df_2019['In which country do you currently reside?'],

                          df_2019['What is your current yearly compensation (approximate $USD)?'])



df_sal = df_sal_temp[(df_sal_temp.index == 'Brazil')| (df_sal_temp.index == 'India') | (df_sal_temp.index == 'Japan') | (df_sal_temp.index == 'Russia') | (df_sal_temp.index == 'United States')

                    |(df_sal_temp.index == 'Canada')| (df_sal_temp.index == 'Germany') | (df_sal_temp.index == "People 's Republic of China")

                    | (df_sal_temp.index == 'United Kingdom')]

titles = 'Countrywise sallary for Data Science'

xlabel= 'Country'

ylabel='Count'

multivariate_plot(df_sal,titles,xlabel,ylabel)
df_sal_temp = pd.crosstab(df_2019['In which country do you currently reside?'],

                          df_2019['What is your current yearly compensation (approximate $USD)?'])



df_sal = df_sal_temp[ (df_sal_temp.index == 'India') | (df_sal_temp.index == 'Japan') | (df_sal_temp.index == 'United States')]

titles = 'Countrywise sallary for Data Science'

xlabel= 'Country'

ylabel='Count'

multivariate_plot(df_sal,titles,xlabel,ylabel)
