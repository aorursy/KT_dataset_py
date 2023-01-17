# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import datetime as dt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

 
import os
df2=pd.read_csv('../input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv')
df2.drop(columns=['address','incident_url','source_url','sources','state_house_district','state_senate_district'], inplace=True)
 

df2.tail()
 
 

#changing the date time
 
def df_return_time_of_month(x):
    if(x<10):
        return 'early'
    elif(x<20):
        return 'mid month'
    else: 
        return 'late' 
df2 = pd.concat([pd.DataFrame([each for each in df2['date'].str.split('-').values.tolist()],
                             columns=['year', 'month', 'day']),df2],axis=1)
 
df2['day_of_week'] = df2.date.apply(lambda x: pd.to_datetime(x).weekday())
df2 = df2.loc[:,~df2.columns.duplicated()]
 
df2=df2.rename_axis('incident') 
df2.head()
 
def df_return(lst):
    dic = {}
    ls = []
    for each in lst:
         ls.append(re.findall(r"[^:|]+", str(each)))
    
    for i, each in enumerate(ls):
        if each == ['nan']:
            dic[i] = {0: 'nan'}
        else:
            dic[i] = dict(key_val for key_val in zip(*([iter(each)] * 2)))
    return pd.DataFrame.from_dict({(i,j): dic[i][j] for i in dic.keys() for j in dic[i].keys()},orient='index')

# gender 
df_gender = df_return(df2['participant_gender']) 
df_gender.index = pd.MultiIndex.from_tuples(df_gender.index)



#age 
df_age=df_return(df2['participant_age'])
df_age.index=pd.MultiIndex.from_tuples(df_age.index)
 

#killed or shot
df_status = df_return(df2['participant_status'])
df_status.index=pd.MultiIndex.from_tuples(df_status.index)


#victim or shooter
df_victimShooter_typ=df_return(df2['participant_type'])
df_victimShooter_typ.index=pd.MultiIndex.from_tuples(df_victimShooter_typ.index)

#gun type 
df_gun_typ = df_return(df2['gun_type']) 
df_gun_typ.index = pd.MultiIndex.from_tuples(df_gun_typ.index)



import matplotlib.pyplot as plt
import scipy
from scipy.stats import ttest_ind
#victim/shooter and gender merge 
df_partiType_gender=pd.merge(df_gender,df_victimShooter_typ,left_index=True,right_index=True,how='left')
df_partiType_gender.columns = ['gender','victim_or_shooter' ]  
 
partiType_gender=pd.crosstab(df_partiType_gender.gender,df_partiType_gender.victim_or_shooter)
partiType_gender.drop('nan',inplace=True)
partiType_gender.drop('nan',axis=1,inplace=True)
partiType_gender.drop('Male, female',axis=0,inplace=True)
 
    
    
    
#motivation 
df_motivation=df_return(df2['participant_relationship'])
df_motivation.index = pd.MultiIndex.from_tuples(df_motivation.index)
 
   #bar graph distrubtuion
#box plot for the mean between female and male 
df_Moti_gender=pd.merge(df_partiType_gender,df_motivation,left_index=True,right_index=True,how='left')
df_Moti_gender.columns = ['gender','victim_or_shooter','motivation']

df_Moti_gender = df_Moti_gender[( df_Moti_gender['motivation'].str.contains('nan', case=False)==False )]
#for bar graph 


fig3 = plt.figure()
 
plt04 = fig3.add_axes([0,0,1,1])
plt_04=sns.countplot(x=('motivation'),data= df_Moti_gender[df_Moti_gender['gender']=='Female'],linewidth=2.5,palette="Set3")
plt_04.set_xticklabels(plt_04.get_xticklabels(), fontsize=8,rotation=40,ha='right')
plt04.set_ylabel("Number of Attackers", fontsize=15)
plt04.set_xlabel("Motivation", fontsize=15)
plt04.legend(loc=9)
fig3.show() 




fig4 = plt.figure()

plt05 = fig4.add_axes([0,0,1,1])
plt_05=sns.countplot(x=('motivation'),data= df_Moti_gender[df_Moti_gender['gender']=='Male'],linewidth=2.5,palette="Set3")
plt05.set_ylabel("Number of Attackers", fontsize=15)
plt_05.set_xticklabels(plt_05.get_xticklabels(), fontsize=8,rotation=40,ha='right')
plt05.set_xlabel("Motivation", fontsize=15)
plt05.legend(loc=9)
fig4.show() 



    
Gender_Percent_Per_Motivation = pd.crosstab(df_Moti_gender[df_Moti_gender['victim_or_shooter'] == 'Subject-Suspect']['motivation'], df_Moti_gender[df_Moti_gender['victim_or_shooter'] == 'Subject-Suspect']['gender']) 
Gender_Percent_Per_Motivation = Gender_Percent_Per_Motivation.div(Gender_Percent_Per_Motivation.sum(), axis=1)*100
 
print(Gender_Percent_Per_Motivation.head(15))    


fig5 = plt.figure()
 
plt06 = fig5.add_axes([0,0,1,1])
colour = ["#7219FF","#00FFAB","#E3085D","#D7B8FF","#F08080","#B3A6A2","#FFF68F","#CCFFFF","#BC9494","#63C2D6","#97A2FF","#D45440","#CCCCDD"]
plt06 = plt.pie(Gender_Percent_Per_Motivation.Female,colors=colour,shadow=True)
#plt.legend(colour,Gender_Percent_Per_Motivation.index, loc="lower right")
fig5.show() 

fig6 = plt.figure()
 
plt07 = fig5.add_axes([0,0,1,1])
colour = ["#7219FF","#00FFAB","#E3085D","#D7B8FF","#F08080","#B3A6A2","#FFF68F","#CCFFFF","#BC9494","#63C2D6","#97A2FF","#D45440","#CCCCDD"]
plt07 = plt.pie(Gender_Percent_Per_Motivation.Male,colors=colour, shadow=True)
#plt.legend(colour,Gender_Percent_Per_Motivation.index, loc="lower right")
fig6.show() 

 

    
#heat chart perhaps for shooters vs victim /female and male 
#age and gender merge for shooters 
df_age_gender=pd.merge(df_partiType_gender,df_age,left_index=True,right_index=True,how='left')
df_age_gender.columns = ['gender','victim_or_shooter','age']
Gender_Per_Age_Shooter = pd.crosstab(df_age_gender[df_age_gender['victim_or_shooter'] == 'Subject-Suspect']['age'], df_age_gender[df_age_gender['victim_or_shooter'] == 'Subject-Suspect']['gender'])

df_age_gender.dropna(inplace=True)
df_victims_age = df_age_gender[df_age_gender['victim_or_shooter'] == 'Subject-Suspect']
df_age_gender= df_age_gender[df_age_gender.age != 'nan']

df_age_gender= df_age_gender[df_age_gender.victim_or_shooter != 'nan']
df_age_gender= df_age_gender[(df_age_gender.gender != 'nan')&(df_age_gender.gender != 'Male, female')]
df_victims_age.age = df_victims_age.age.astype(float)
df_age_gender.age = df_age_gender.age.astype(float)
df_age_gender= df_age_gender[df_age_gender.age < 120 ]

 
#bar graph distrubtuion

fig = plt.figure()
 
plt01 = fig.add_axes([0,0,1,1])
plt01 = sns.barplot(x=('gender'),y=('age'),data = df_victims_age,linewidth=2.5,palette="Set3") 
plt01.set_ylabel("Age", fontsize=15)
plt01.set_xlabel("Group Studied", fontsize=15)
plt01.legend(loc=9)
fig.show() 

df_male_attacker = df_age_gender[(df_age_gender.gender != 'Female')&(df_age_gender['victim_or_shooter'] == 'Subject-Suspect')]
df_female_attacker = df_age_gender[(df_age_gender.gender != 'Male')&(df_age_gender['victim_or_shooter'] == 'Subject-Suspect')]




fig2=plt.figure()
plt02=fig2.add_axes([0,0,1,1])
plt03=fig2.add_axes([0,0,1,1])
plt02=sns.kdeplot(df_male_attacker['age'],label="Male",color="green")
plt02.legend(loc=1)
plt02.set_ylabel("Probability Density", fontsize=15)
plt02.set_xlabel("Age", fontsize=15)
plt03=sns.kdeplot(df_female_attacker['age'],label ="Female",color="yellow")
plt03.legend(loc=1)
fig2.show() 
 
#box plot for the mean between female and male 

df_age_victim=pd.merge(df_age,df_victimShooter_typ,left_index=True,right_index=True,how='left')
df_age_victim.columns = ['Age','Victims' ]  
Age_Victim=pd.crosstab(df_age_victim.Age,df_age_victim[df_age_victim['Victims']=='Victim'].Victims)

Gender_Per_Age_Victim = pd.crosstab(df_age_gender[df_age_gender['victim_or_shooter'] == 'Victim']['age'], df_age_gender[df_age_gender['victim_or_shooter'] == 'Victim']['gender'])


print( df_male_attacker.head())
print("female age mean : " +str( df_female_attacker['age'].mean()))
print("male age mean : " + (str(df_male_attacker['age'].mean())))
ttest_ind( df_female_attacker['age'],df_male_attacker['age'],equal_var = False)

 

import scipy.stats as stats
df_new= pd.crosstab(df_Moti_gender[df_Moti_gender['victim_or_shooter'] == 'Subject-Suspect']['motivation'], df_Moti_gender[df_Moti_gender['victim_or_shooter'] == 'Subject-Suspect']['gender'],margins=True) 
print(df_new.head(15))
df_new.columns=["Female","Male","Row_Total"]
df_new.row=["Motivation","Acquitance","Armed Robbery","Co-worker","Drive by - Random victims","Family","Friends",
                "Gang vs Gang","Home Invasion - Perp Does Not Know Victim","Home Invasion - Perp Knows Victim","Mass shooting - Perp Knows Victims","Mass shooting - Random victims",
               "Neighbor","Significant others - current or former","Column_Total"]
observed=df_new.iloc[0:13,0:2]
 
expected= np.outer(df_new["Row_Total"][0:14],df_new.iloc[13][0:2])/19343
expected=pd.DataFrame(expected)

expected.columns=["Female","Male" ]
expected.index=["Motivation","Acquitance","Armed Robbery","Co-worker","Drive by - Random victims","Family","Friends",
                "Gang vs Gang","Home Invasion - Perp Does Not Know Victim","Home Invasion - Perp Knows Victim","Mass shooting - Perp Knows Victims","Mass shooting - Random victims",
               "Neighbor","Significant others - current or former"]
print(expected.head())

chi_squared_stat=(((observed-expected)**2)/expected).sum().sum()
# we call sum twice because once we want to get the column sum 
# then we want to add the coulmn sum together
print(chi_squared_stat)
crit=stats.chi2.ppf(q=0.95,df=12)
print("critical p")
print(crit)
p_value= 1 - stats.chi2.cdf(x=chi_squared_stat,df=12)
print("pvalue")
print(p_value)
stats.chi2_contingency(observed=observed)
#gun type and gender merge
df_gun_gender = pd.merge(df_gun_typ, df_gender, left_index=True, right_index=True, how='left')
df_gun_gender.columns = ['gun_type', 'gender']
gun_sex = pd.crosstab(df_gun_gender.gun_type, df_gun_gender.gender)
gun_sex.drop('nan', inplace=True)
gun_sex.drop('nan', axis= 1, inplace=True)
gun_sex.drop('Unknown', inplace=True)
 


layout = dict(title = "Type of Guns Used by each Sex", xaxis = dict(title = 'Type'), yaxis = dict(title = 'Count')) 
trace= []
for i, each in enumerate(gun_sex):
    trace.append(go.Bar(x = gun_sex.index, y =gun_sex[each], name= each ))
data = go.Data(trace)
fig = go.Figure(data= data, layout=layout)
#py.offline.iplot(fig)
 

#df.to_csv
# Any results you write to the current directory are saved as output.
 

df2.head()
 
df_gender = df_gender.reindex(df_gender.index.rename(['incident','person']))
 
df_victimShooter_typ=df_victimShooter_typ.reindex(df_victimShooter_typ.index.rename(['incident','person'])) 
df_gender_victim=pd.merge(df_gender,df_victimShooter_typ,how='inner',on=['incident','person'])
df_gender_victim.head()
df_gender_victim.columns=['Gender','Victim_or_suspect']


df_gender_year = pd.merge(df_gender_victim, df2[['year','month','day','day_of_week']],how='left', on=['incident'])

df_gender_year.columns=['Gender','Victim_or_suspect','year','month','day','day_of_week']


df_gender_year_attackers = df_gender_year[df_gender_year['Victim_or_suspect']=='Subject-Suspect']
df_gender_year_victims = df_gender_year[df_gender_year['Victim_or_suspect']=='Victim']
df_female_attacker = df_gender_year_attackers[df_gender_year_attackers['Gender']=='Female']
df_male_attacker = df_gender_year_attackers[df_gender_year_attackers['Gender']=='Male']
df_female_victims = df_gender_year_victims[df_gender_year_victims['Gender']=='Female']
df_male_victims = df_gender_year_victims[df_gender_year_victims['Gender']=='Male']

heatPlot_female_attackers= df_female_attacker.groupby(["year", "month"]).size().reset_index(name="Number of Attackers")
heatPlot_male_attackers= df_male_attacker.groupby(["year", "month"]).size().reset_index(name="Number of Attackers")
heatPlot_female_victims= df_female_victims.groupby(["year", "month"]).size().reset_index(name="Number of Victims")
heatPlot_male_victims= df_male_victims.groupby(["year", "month"]).size().reset_index(name="Number of Victims")
 


fig_time_heat_fa =plt.figure() 
plt10 = fig_time_heat_fa.add_axes([0,0,1,1])
plt10 = sns.heatmap(heatPlot_female_attackers.pivot_table(columns='year',index='month',values='Number of Attackers'), cmap="YlGnBu")
plt10.set_title("Number of Female Attackers")    
plt10.legend(loc=0)
plt10.set_xlabel("Year", fontsize=15)
plt10.set_ylabel("Month", fontsize=15)
fig_time_heat_fa.show() 

fig_time_heat_ma =plt.figure() 
plt11 = fig_time_heat_ma.add_axes([0,0,1,1])
plt11 = sns.heatmap(heatPlot_male_attackers.pivot_table(columns='year',index='month',values='Number of Attackers'), cmap="YlGnBu")
plt11.set_title("Number of Male Attackers") 
plt11.legend(loc=0)
plt11.set_xlabel("Year", fontsize=15)
plt11.set_ylabel("Month", fontsize=15)
fig_time_heat_ma.show()

fig_time_heat_fv =plt.figure() 
plt12=fig_time_heat_fv.add_axes([0,0,1,1])
plt12 = sns.heatmap(heatPlot_female_victims.pivot_table(columns='year',index='month',values='Number of Victims'), cmap="YlGnBu")
plt12.set_title("Number of Female Victims") 
plt12.legend(loc=0)
plt12.set_xlabel("Year", fontsize=15)
plt12.set_ylabel("Month", fontsize=15)
fig_time_heat_mv =plt.figure() 

plt13=fig_time_heat_mv.add_axes([0,0,1,1])
plt13 = sns.heatmap(heatPlot_male_victims.pivot_table(columns='year',index='month',values='Number of Victims'), cmap="YlGnBu")
plt13.set_title("Number of Male Victims") 
plt13.legend(loc=0)
plt13.set_xlabel("Year", fontsize=15)
plt13.set_ylabel("Month", fontsize=15)

 


import plotly 
import pandas
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot  
# day of the week 
#gender_day_of_week = pd.crosstab(df_gender_year[df_gender_year['Victim_or_suspect']=='Subject-Suspect']['day_of_week'],
                             #    df_gender_year[df_gender_year['Victim_or_suspect']=='Subject-Suspect']['Gender'])
df_gender_year = df_gender_year[df_gender_year['Victim_or_suspect']=='Subject-Suspect']
 
#df_gender_year.drop(['month','year','day'],axis=1)
gender_day_of_week =  df_gender_year.groupby(['day_of_week','Gender','year','month']).size().reset_index(name="Number of Victims") 
print(gender_day_of_week.head())
 
fig1week1 = plt.figure()
#we need to change this so it's the weekly's average , not sum  
plt_week1 = fig1week1.add_axes([0,0,1,1])

plt_week1 = sns.boxplot(x="day_of_week", y="Number of Victims", data=gender_day_of_week[gender_day_of_week['Gender']=='Female'], palette="Set3")
plt_week1.set_xticklabels(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], fontsize=8,rotation=40,ha='right')
plt_week1.set_ylabel("Number of Attackers", fontsize=12)
plt_week1.set_xlabel("Day of the Week", fontsize=12)
plt_week1.set_title("Total Number of Female Attackers Per week ")
plt_week1.legend(loc=9)
fig1week1.show() 


fig1week2 = plt.figure()
#we need to change this so it's the weekly's average , not sum  
plt_week2 = fig1week2.add_axes([0,0,1,1])
plt_week2 = sns.boxplot(x="day_of_week", y="Number of Victims", data=gender_day_of_week[gender_day_of_week['Gender']=='Male'], palette="Set3")
plt_week2.set_xticklabels(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], fontsize=8,rotation=40,ha='right')
plt_week2.set_ylabel("Number of Attackers", fontsize=12)
plt_week2.set_xlabel("Day of the Week", fontsize=12)
plt_week2.set_title("Total Number of Male Attackers Per Year ")
plt_week2.legend(loc=9)
fig1week2.show() 



fig1year1 = plt.figure()
#we need to change this so it's the weekly's average , not sum  , compare it by month and year 
plt_year1 = fig1year1 .add_axes([0,0,1,1])
plt_year1 = sns.countplot(x=('year'),data= df_female_attacker[df_female_attacker['Gender']=='Female'],linewidth=2.5,palette="Set3")
plt_year1.set_xticklabels(['2013','2014','2015','2016','2017','2018'], fontsize=8,rotation=40,ha='right')
plt_year1.set_ylabel("Number of Attackers", fontsize=12)
plt_year1.set_xlabel("Year", fontsize=12)
plt_year1.set_title("Total Number of Female Attackers Per Year")
plt_year1.legend(loc=9)
fig1year1.show() 

fig1year2 = plt.figure()
#we need to change this so it's the weekly's average , not sum  
plt_year2 = fig1year2.add_axes([0,0,1,1])
plt_year2 = sns.countplot(x=('year'),data= df_male_attacker[df_male_attacker['Gender']=='Male'],linewidth=2.5,palette="Set3")
plt_year2.set_xticklabels(['2013','2014','2015','2016','2017','2018'], fontsize=8,rotation=40,ha='right')
plt_year2.set_ylabel("Number of Attackers", fontsize=12)
plt_year2.set_xlabel("Year", fontsize=12)
plt_year2.set_title("Total Number of Male Attackers Per week ")
plt_year2.legend(loc=9)
fig1year2.show() 


#
#
#
#

num_of_male_counts=df_male_attacker['Gender'].count()
num_of_male_counts_2013 = df_male_attacker[df_male_attacker['year']=='2013']['Gender'].count()
num_of_male_counts_2014 = df_male_attacker[df_male_attacker['year']=='2014']['Gender'].count()
num_of_male_counts_2015 = df_male_attacker[df_male_attacker['year']=='2015']['Gender'].count()
num_of_male_counts_2016 = df_male_attacker[df_male_attacker['year']=='2016']['Gender'].count()
num_of_male_counts_2017 = df_male_attacker[df_male_attacker['year']=='2017']['Gender'].count()
num_of_male_counts_2018 = df_male_attacker[df_male_attacker['year']=='2018']['Gender'].count()
#print("total number of male attackers in 2013 :"+ str(num_of_male_counts_2013))
#print("total number of male attackers in 2014 :"+ str(num_of_male_counts_2014))
#print("total number of male attackers in 2015 :"+ str(num_of_male_counts_2015))
#print("total number of male attackers in 2016 :"+ str(num_of_male_counts_2016))
#print("total number of male attackers in 2017 :"+ str(num_of_male_counts_2017))
#print("total number of male attackers in 2018 :"+ str(num_of_male_counts_2018))

num_of_female_counts_2013 = df_female_attacker[df_female_attacker['year']=='2013']['Gender'].count()
num_of_female_counts_2014 = df_female_attacker[df_female_attacker['year']=='2014']['Gender'].count()
num_of_female_counts_2015 = df_female_attacker[df_female_attacker['year']=='2015']['Gender'].count()
num_of_female_counts_2016 = df_female_attacker[df_female_attacker['year']=='2016']['Gender'].count()
num_of_female_counts_2017 = df_female_attacker[df_female_attacker['year']=='2017']['Gender'].count()
num_of_female_counts_2018 = df_female_attacker[df_female_attacker['year']=='2018']['Gender'].count()
#print("total number of female attackers in 2013 :"+ str(num_of_female_counts_2013))
#print("total number of female attackers in 2014 :"+ str(num_of_female_counts_2014))
#print("total number of female attackers in 2015 :"+ str(num_of_female_counts_2015))
#print("total number of female attackers in 2016 :"+ str(num_of_female_counts_2016))
#print("total number of female attackers in 2017 :"+ str(num_of_female_counts_2017))
#print("total number of female attackers in 2018 :"+ str(num_of_female_counts_2018))


num_of_female_counts=df_female_attacker['Gender'].count()

print("total number of male attackers :"+ str(num_of_male_counts))
print("total number of female attackers : "+ str(num_of_female_counts))



#sns.boxplot(x='day_of_week',y='Number of Attackers',data=gender_day_of_week, hue='Gender') 
    
 
 #do women/male tend to kill/harm more?
#do this if we have more time 
 

 
df_time_series = pd.merge(df_gender_victim, df2[['date']], how='left', on=['incident'])
df_time_series.columns=['Gender','Victim_or_suspect','date']

print(df_time_series.head(10))
df_time_female = df_time_series [(df_time_series['Gender']=='Female')&(df_time_series['Victim_or_suspect']=='Subject-Suspect')]
df_time_male = df_time_series [(df_time_series['Gender']=='Male')&(df_time_series['Victim_or_suspect']=='Subject-Suspect')]

df_time_female= df_time_female.groupby(["date"]).size().reset_index(name="Number of Attackers")
df_time_male= df_time_male.groupby(["date"]).size().reset_index(name="Number of Attackers")
 

fig1time1 = plt.figure()
#we need to change this so it's the weekly's average , not sum  
plt_time1 = fig1time1.add_axes([0,0,1,1])
plt_time1 = plt.plot(df_time_female.date, df_time_female["Number of Attackers"],linewidth=0.5)
# plt_time1.set_xticklabels(['2013','2014','2015','2016','2017','2018'], fontsize=8,rotation=40,ha='right')
#plt_time1.set_ylabel("Number of Attackers", fontsize=12)
#plt_time1.set_xlabel("Year", fontsize=12) 
fig1time1.show() 



fig1time2 = plt.figure()
#we need to change this so it's the weekly's average , not sum  
plt_time2 = fig1time2.add_axes([0,0,1,1])
plt_time2 = plt.plot(df_time_male.date, df_time_male["Number of Attackers"],linewidth=0.5)
# plt_time1.set_xticklabels(['2013','2014','2015','2016','2017','2018'], fontsize=8,rotation=40,ha='right')
#plt_time1.set_ylabel("Number of Attackers", fontsize=12)
#plt_time1.set_xlabel("Year", fontsize=12) 
fig1time2.show() 





df_gender_location = pd.merge(df_gender_victim, df2[['state']],
                       how='left', on=['incident'])
df_gender_location.columns=['Gender','Victim_or_suspect','state']

#remember to cross it again later, also remember to drop everything that is victim  
 
df_Moti_gender = df_Moti_gender.reindex(df_Moti_gender.index.rename(['incident','person']))

df_age= df_age.reindex(df_age.index.rename(['incident','person']))
df_age.columns=["age"]

df_moti_corr =pd.merge(df_Moti_gender, df2[['state','year','month','day_of_week']],how='left', on=['incident'])

df_moti_corr.head()
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df_corr =pd.merge(df_age,df_moti_corr,how='left', on=['incident'])
df_corr.dropna(inplace=True)


y=df_corr.motivation
X_train, X_test, y_train, y_test = train_test_split(df_corr, y, test_size=0.2)

logreg=LogisticRegression() 
#coeff_df=pd.DataFrame(logreg.column.delete(0))
#coeff_df.column=['Feature']
#coeff_df["Correlation"]



