# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from scipy.stats import norm 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import folium      #  folium libraries

from   folium.plugins import MarkerCluster

import os

print(os.listdir("../input"))

import datetime

%matplotlib inline

# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# init_notebook_mode(connected=False)

# # import chart_studio.plotly

# import cufflinks as cf

# cf.go_offline()



# PLOTLY NOT WORKING IN KAGGLE !!  :( :(  

# So, commenting it out and the code of world geogrphay created using it :( :( 
# Any results you write to the current directory are saved as output.

# victims of rape DataFrame

rape_victim = pd.read_csv('../input/crime-in-india/20_Victims_of_rape.csv',na_filter = 'False')

rape_victims_2016 = pd.read_csv('../input/2016crimedata/Rape_Victims_Table_3A.3_2016.csv',na_filter = 'False',encoding = 'unicode_escape') # data has utf8 characters



# let's see how the data is structured

rape_victim.sample(5).style.set_table_styles(

[{'selector': 'tr:hover',

  'props': [('background-color', 'yellow')]}]

)
rape_victims_2016.sample(5).style.set_table_styles(

[{'selector': 'tr:hover',

  'props': [('background-color', 'yellow')]}]

)
rape_victim.columns
rape_victim.fillna('')

rape_victim.isnull().sum().sum()
total_rape = rape_victim[rape_victim['Subgroup'] == 'Total Rape Victims']

cp_rape_victim = rape_victim.copy()
total_rape.describe().style.set_table_styles(

[{'selector': 'tr:hover',

  'props': [('background-color', 'yellow')]}]

)
total_rape.info()
total_rape.Year = pd.to_datetime(total_rape['Year'], format ='%Y').dt.strftime('%Y')

total_rape.loc[:,'Total_Rape_per_Year'] = total_rape.groupby('Year')['Victims_of_Rape_Total'].transform('sum')

plot_total_rape  = total_rape.drop_duplicates('Year', keep = 'first', inplace = False)

#plot_total_rape.drop(['Area_Name'], axis = 1,inplace = False)

#plot_total_rape.columns

plot_total_rape.sample()
# Creating a Countplot

plt.figure(figsize=(14,10))

#plot_total_rape.plot(legend=False)

x = plot_total_rape['Year']

y = plot_total_rape['Total_Rape_per_Year']

plt.plot(x,y)

plt.title('Number of rapes per year in all of India for the decade (2001 - 2010)',color = 'blue',fontsize=14)

plt.xlabel('Year')

plt.ylabel('Total_Rape_per_Year')

plt.grid(True)

plt.show()
# Plotting Victims Juvenile vs Adults

#total_rape_minor = plot_total_area_rape['Victims_Upto_10_Yrs'].sum() + plot_total_area_rape['Victims_Between_10-14_Yrs'].sum() +plot_total_area_rape['Victims_Between_14-18_Yrs'].sum()

#total_rape_adults = plot_total_area_rape['Victims_Between_18-30_Yrs'].sum() + plot_total_area_rape['Victims_Between_30-50_Yrs'].sum() +plot_total_area_rape['Victims_Above_50_Yrs'].sum()



total_rape_minor = plot_total_rape['Victims_Upto_10_Yrs'].sum() + plot_total_rape['Victims_Between_10-14_Yrs'].sum() +plot_total_rape['Victims_Between_14-18_Yrs'].sum()

total_rape_adults = plot_total_rape['Victims_Between_18-30_Yrs'].sum() + plot_total_rape['Victims_Between_30-50_Yrs'].sum() +plot_total_rape['Victims_Above_50_Yrs'].sum()



data = [total_rape_minor,total_rape_adults ] 

index =['Adults Victims', 'Minor Victims']

data_df = pd.DataFrame(data, columns = ['age_level'], index = index)

#print(data)

plt.figure(figsize=(4,3), edgecolor='black')

ax = data_df.plot.bar(rot=0)

plt.title("Minor vs Adults - Victims",color = 'royalblue',fontsize=14)
cp_rape_victim.sample()

cp_total_rape = cp_rape_victim[rape_victim['Subgroup'] == 'Total Rape Victims']

cp_total_rape.Year = pd.to_datetime(cp_total_rape['Year'], format ='%Y').dt.strftime('%Y')

cp_total_rape.loc[:,'Total_Rape_per_Area'] = cp_total_rape.groupby('Area_Name')['Victims_of_Rape_Total'].transform('sum')

plot_total_area_rape = cp_total_rape.drop_duplicates('Area_Name')

#cp_total_rape
# 2001 statistics 

population_2001 = pd.read_csv('../input/census2001/all.csv', na_filter = False)

population_2001.replace({

                'UP' : 'Uttar Pradesh',

                'TN' : 'Tamil Nadu',

                'MP' : 'Madhya Pradesh',

                'D_N_H' : 'Dadra & Nagar Haveli',

                'HP' : 'Himachal Pradesh',

                'WB' : 'West Bengal',

                'CG' : 'Chhattisgarh',

                'JK' : 'Jammu & Kashmir',

                'AN' : 'Andaman & Nicobar Islands',

                'D_D' : 'Daman & Diu'    

},inplace=True

)



population_2001.loc[:,'State_Popu'] = population_2001.groupby('State')['Persons'].transform('sum')



population_2001_cp = population_2001.drop_duplicates('State',keep = 'first')

population_2001_use = population_2001_cp[['State','State_Popu']]

population_2001_use.reset_index()

population_2001_use.loc[:,'Year'] = '2001'

population_2002 = population_2001_use.copy()



def rule(year, pop, df):

        df['Year'] = 'year'

        df['State_Popu'] = df['State_Popu'] * pop

        return df

year_map = {'2001': '2002', '2002': '2003' , '2003': '2004', '2004': '2005', '2005': '2006', '2006': '2007', '2007': '2008', '2008': '2009', '2009': '2010'   }



population_2002_use = population_2001_use.copy()

population_2002_use['Year'] = population_2001_use['Year'].map(year_map)

population_2002_use['State_Popu'] =  population_2002_use['State_Popu']*1.017  # India's population increase rate applied on each state.

population_2002_use.State_Popu = population_2002_use.State_Popu.astype(int)



population_2003_use = population_2002_use.copy()

population_2003_use['Year'] = population_2002_use['Year'].map(year_map)

population_2003_use['State_Popu'] = population_2002_use['State_Popu']* 1.017 # India's population increase rate applied on each state.

population_2003_use.State_Popu = population_2002_use.State_Popu.astype(int)





population_2004_use = population_2003_use.copy()

population_2004_use['Year'] = population_2003_use['Year'].map(year_map)

population_2004_use['State_Popu'] =  population_2003_use['State_Popu']*1.016  # India's population increase rate applied on each state.

population_2004_use.State_Popu = population_2003_use.State_Popu.astype(int)



population_2005_use = population_2004_use.copy()

population_2005_use['Year'] = population_2004_use['Year'].map(year_map)

population_2005_use['State_Popu'] = population_2004_use['State_Popu']* 1.016 # India's population increase rate applied on each state.

population_2005_use.State_Popu = population_2004_use.State_Popu.astype(int)



population_2006_use = population_2005_use.copy()

population_2006_use['Year'] = population_2005_use['Year'].map(year_map)

population_2006_use['State_Popu'] =  population_2005_use['State_Popu']*1.015  # India's population increase rate applied on each state.

population_2006_use.State_Popu = population_2005_use.State_Popu.astype(int)



population_2007_use = population_2006_use.copy()

population_2007_use['Year'] = population_2006_use['Year'].map(year_map)

population_2007_use['State_Popu'] = population_2006_use['State_Popu']* 1.015 # India's population increase rate applied on each state.

population_2007_use.State_Popu = population_2006_use.State_Popu.astype(int)





population_2008_use = population_2007_use.copy()

population_2008_use['Year'] = population_2007_use['Year'].map(year_map)

population_2008_use['State_Popu'] =  population_2007_use['State_Popu']*1.015  # India's population increase rate applied on each state.

population_2008_use.State_Popu = population_2007_use.State_Popu.astype(int)



population_2009_use = population_2008_use.copy()

population_2009_use['Year'] = population_2008_use['Year'].map(year_map)

population_2009_use['State_Popu'] = population_2008_use['State_Popu']* 1.014 # India's population increase rate applied on each state.

population_2009_use.State_Popu = population_2008_use.State_Popu.astype(int)



population_2010_use = population_2009_use.copy()

population_2010_use['Year'] = population_2009_use['Year'].map(year_map)

population_2010_use['State_Popu'] = population_2009_use['State_Popu']* 1.014 # India's population increase rate applied on each state.

population_2010_use.State_Popu = population_2009_use.State_Popu.astype(int)





population_2000_decade_use = [population_2001_use, population_2002_use, population_2003_use, population_2004_use, population_2005_use, population_2006_use , population_2007_use, population_2008_use, population_2009_use, population_2010_use  ]



population_2000_decade = pd.concat(population_2000_decade_use)



population_2000_decade.sample(5)





merge_total_rape = pd.merge(population_2000_decade,cp_total_rape, left_on = ['State', 'Year'], right_on= ['Area_Name', 'Year'] , how = 'inner' )

merge_total_rape.sample(5)

merge_total_rape.loc[:,'Victim_vs_Population'] =  merge_total_rape['Total_Rape_per_Area']/merge_total_rape['State_Popu']

merge_total_rape.sample()
# heat map plotting 



in_data2 = total_rape.pivot("Area_Name","Year","Victims_of_Rape_Total" )



plt.figure(figsize=(14, 10)) 



plt.yticks(rotation=1) 

ax = sns.heatmap(in_data2,cmap="YlGnBu", linewidths=.5)

plt.title("State wise Count Distribution of Crime Rape",color = 'blue',fontsize=14)
# heat map plotting 



in_data3 = merge_total_rape.pivot("Area_Name","Year","Victim_vs_Population" )



plt.figure(figsize=(14, 10)) 



plt.yticks(rotation=1) 

ax = sns.heatmap(in_data3,cmap="YlGnBu", linewidths=.5)

plt.title("State wise Distribution of Crime Rape as per Population",color = 'blue',fontsize=14)
rape_offender = pd.read_csv('../input/21-offenders-known-to-the-victimcsv/21_Offenders_known_to_the_victim.csv',na_filter = 'False')

rape_offender.head()

rape_offender.fillna('')

rape_offender.Year = pd.to_datetime(rape_offender['Year'], format ='%Y').dt.strftime('%Y')

rape_offender.isnull().sum().sum()
joined_df = pd.merge(rape_offender, total_rape, on = ['Area_Name', 'Year'], how = 'inner'  )

joined_df['Known_to_Offender_Percent'] = joined_df.No_of_Cases_in_which_offenders_were_known_to_the_Victims/ joined_df.Rape_Cases_Reported

joined_df.replace([np.inf, -np.inf], np.nan)

joined_df  = joined_df.dropna(axis=0, subset = ['Known_to_Offender_Percent'])

joined_df.isnull().sum().sum()
joined_df1 = joined_df[['Area_Name', 'Year','No_of_Cases_in_which_offenders_were_known_to_the_Victims', 'Rape_Cases_Reported']].copy()



joined_df2 = joined_df1.replace([np.inf, -np.inf], np.nan).dropna(axis=0)



joined_df2['Sum_known_offenders'] = joined_df2.groupby('Year')['No_of_Cases_in_which_offenders_were_known_to_the_Victims'].transform('sum')

joined_df2['Sum_total_rapes'] = joined_df2.groupby('Year')['Rape_Cases_Reported'].transform('sum')

plot_joined_df2 = joined_df2.drop_duplicates('Year', keep = 'first', inplace = False)

plot_joined_df2 = plot_joined_df2.sort_values('Year')
#plot based group by on Year

fig, ax = plt.subplots(figsize=(14,10))

sns.lineplot(x=plot_joined_df2.Year, 

             y=plot_joined_df2.Sum_total_rapes, 

             color='royalblue',

             ax=ax) 



sns.lineplot(x=plot_joined_df2.Year, 

             y=plot_joined_df2.Sum_known_offenders,

             color='seagreen',

             ax=ax)

   

ax.legend(['total_rapes' , 'offenders_known'], facecolor='y')

ax.set(ylim=(1000, 25000))



plt.grid(True)

plt.title("How Often Victims and Offenders are Related",color = 'blue',fontsize=14)

plt.show()
joined_df3 = joined_df1.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

joined_df3['Sum_known_offenders'] = joined_df3.groupby('Area_Name')['No_of_Cases_in_which_offenders_were_known_to_the_Victims'].transform('sum')

joined_df3['Sum_total_rapes'] = joined_df3.groupby('Area_Name')['Rape_Cases_Reported'].transform('sum')



plot_joined_df3 = joined_df3.drop_duplicates('Area_Name', keep = 'first', inplace = False)

plot_joined_df3 = plot_joined_df3.sort_values('Area_Name')

plot_joined_df3.drop(['Year'], axis =1, inplace =True)

#plot_joined_df3
sns.set(style="whitegrid")



f, ax = plt.subplots(figsize=(14,10))





sns.set_color_codes("pastel")

sns.barplot(x='Sum_total_rapes', y='Area_Name', data=plot_joined_df3,

            label="Total Rapes", color="royalblue", edgecolor='black')





sns.set_color_codes("muted")

sns.barplot(x='Sum_known_offenders', y='Area_Name', data=plot_joined_df3,

            label="Involving Known Offenders", color="seagreen", edgecolor='black')





ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 35000), ylabel="",

       xlabel="Total Rapes vs Victims known to Offenders")

plt.title("How Often Victims and Offenders are Related acorss India ?",color = 'blue',fontsize=14)



sns.despine(left=True, bottom=True)
census_2011 = pd.read_csv('../input/india-census/india-districts-census-2011.csv', na_filter = False)

census_2011.fillna('')

pd.set_option('display.max_columns', None)  

pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)



census_2011.sample(5).style.set_table_styles(

[{'selector': 'tr:hover',

  'props': [('background-color', 'yellow')]}]

)
merge_total_rape_2010 =  merge_total_rape.loc[merge_total_rape['Year'] == '2010']

merge_total_rape_2010.sample()
census_2011.loc[:,'Total_State_Population'] = census_2011.groupby('State name')['Population'].transform('sum')

census_2011_use = census_2011[['State name', 'Total_State_Population', 'Male', 'Female','Literate', 'Male_Literate', 'Female_Literate','Literate_Education','Illiterate_Education','SC','ST','Hindus',  'Muslims', 'Christians', 'Sikhs' ,'Buddhists' ,'Jains',  'Total_Education' , 'Workers', 'Male_Workers','Female_Workers', 'Marginal_Workers', 'Non_Workers','Rural_Households', 'Urban_Households', 'Households', 'Having_bathing_facility_Total_Households','Having_latrine_facility_within_the_premises_Total_Households']]

census_2011_use.loc[:,'Area_Name'] = census_2011_use['State name'].apply(lambda x: x.capitalize()) 

census_2011_use.sample(5)



#plot_joined_df3

merged_df3_2011 = pd.merge(merge_total_rape_2010, census_2011_use, left_on ='State', right_on = 'Area_Name', how = 'inner')



merged_df3_2011.loc[:,'Victim_vs_Population_2010'] =  merged_df3_2011['Total_Rape_per_Area']/merged_df3_2011['Total_State_Population']

#merged_df3_2011.sample(5)
merged_df3_2011_use = merged_df3_2011.drop_duplicates('State', keep = 'first', inplace = False)

merged_df3_2011_use.sample(5).style.set_table_styles(

[{'selector': 'tr:hover',

  'props': [('background-color', 'yellow')]}]

)
from numpy import cov

from scipy.stats import pearsonr

from scipy.stats import spearmanr
# relationship between  Rape/ Population of Area with amount of female literacy

cov_f_lit = cov(merged_df3_2011_use['Victim_vs_Population_2010'] , merged_df3_2011_use['Female_Literate'])

print("\033[1;34;47m Covariance between female literacy with rape/ population - \n")

print(cov_f_lit)

# negaive co-relation
# Female literacy is noteable relationship with rape rate per area.

corr_f_lit_p , _ = pearsonr(merged_df3_2011_use['Victim_vs_Population'] , merged_df3_2011_use['Female_Literate'])

print("\033[1;34;47mPearsons correlation between femlae literacy with rape/ population: %.3f" % corr_f_lit_p)

# Female literacy is noteable relationship with rape rate per area.

corr_f_lit_s, _ = spearmanr(merged_df3_2011_use['Victim_vs_Population'] , merged_df3_2011_use['Female_Literate'])

print('Spearmans correlation between female literacy with rape/ population: %.3f' % corr_f_lit_s)
# relationship between  Rape/ Population of Area with non workers in an area indicating unemployment.

cov_non_wrk = cov(merged_df3_2011_use['Victim_vs_Population'] , merged_df3_2011_use['Non_Workers'])

print("\033[1;34;47m Covariance between factors of non-workers with rape/ population- \n")

print(cov_non_wrk)
# Unemployment has noteable relationship with rape rate per area.

cov_non_wrk_p, _ = pearsonr(merged_df3_2011_use['Victim_vs_Population'] , merged_df3_2011_use['Non_Workers'])

print("\033[1;34;47m Pearsons correlation between non-workers with rape/ population : %.3f" % cov_non_wrk_p)

# Unemployment has noteable relationship with rape rate per area.

cov_non_wrk_s, _ = spearmanr(merged_df3_2011_use['Victim_vs_Population'] , merged_df3_2011_use['Non_Workers'])

print('Spearmans correlation between non-workers with rape/ population : %.3f' % cov_non_wrk_s)
# Does Literacy rate impact Rapes ?

lit_merged_df3_2011 = merged_df3_2011[['Victim_vs_Population_2010' ,'State name', 'Total_State_Population', 'Male', 'Female','Literate', 'Male_Literate', 'Female_Literate','Literate_Education','Illiterate_Education', 'Total_Education']]

lit_merged_df3_2011_use = lit_merged_df3_2011.drop_duplicates('State name', keep = 'first', inplace = False)

corr_lit_merged_df3_2011 = lit_merged_df3_2011_use.corr()

corr_lit_merged_df3_2011

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corr_lit_merged_df3_2011, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
# Does Working male/female ratio impact Rapes ?

wrk_merged_df3_2011 = merged_df3_2011[['Victim_vs_Population_2010' ,'State name', 'Total_State_Population', 'Male', 'Female','Workers','Male_Workers', 'Female_Workers']]

wrk_merged_df3_2011_use = wrk_merged_df3_2011.drop_duplicates('State name', keep = 'first', inplace = False)

corr_wrk_merged_df3_2011 = wrk_merged_df3_2011.corr()



f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corr_wrk_merged_df3_2011, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
from numpy import mean

from numpy import std

from numpy.random import randn

from numpy.random import seed

from matplotlib import pyplot


pyplot.scatter(merged_df3_2011_use['Total_Rape_per_Area'] , merged_df3_2011_use['Having_latrine_facility_within_the_premises_Total_Households'])

# pyplot.scatter(merged_df3_2011_use['Total_State_Population'] , merged_df3_2011_use['Having_bathing_facility_Total_Households'])

pyplot.show()
# Does quality of household hold any impact ?

hou_merged_df3_2011 = merged_df3_2011[['Victim_vs_Population_2010' ,'State name', 'Total_State_Population', 'Households', 'Rural_Households', 'Urban_Households', 'Having_bathing_facility_Total_Households','Having_latrine_facility_within_the_premises_Total_Households']]

hou_merged_df3_2011_use = hou_merged_df3_2011.drop_duplicates('State name', keep = 'first', inplace = False)

corr_hou_merged_df3_2011_use = hou_merged_df3_2011_use.corr()



f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corr_hou_merged_df3_2011_use, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
forced_rape = pd.read_csv('../input/crime-in-india/39_Specific_purpose_of_kidnapping_and_abduction.csv',na_filter = 'False')

forced_rape.head(3)

""" The 'relevant data' is under Group Name -"Kidnap - For Illicit Intercourse" and Sub group name - "04. For Illicit Intercourse". How ?

The data under 'Group Name' is all kinds of kidnapping data - reasons ranging from begging, sale, marriage, prostitution, etc. we can safely 

say that, while we talk about 'Rape' as a singular crime it has to only for single or multple intercourses without the consent of victim. 

Thus, from the above data set, data is filtered out on Group Name -"Kidnap - For Illicit Intercourse". Under this, only 1 sub group is present.

The next column 'K_A_Cases_Reported' talks about reported cases of kidnapping and abduction all over India for the decade.

"""

forced_rape.fillna('')





forced_rape = forced_rape[(forced_rape['Group_Name'] == 'Kidnap - For Illicit Intercourse') ] #| (forced_rape['Group_Name'] == 'Kidnap - For Prostitution')]

#forced_rape



#considering rapes of ALL genders.

total_forced_rapes = forced_rape['K_A_Cases_Reported'].sum()

x = total_forced_rapes.astype('int64')

print("\033[1;34;47m Total Kidnapping for the purpose of rape in entire India -")

print(x)

total_rape_plot = plot_total_area_rape.drop_duplicates('Area_Name')

y = total_rape_plot['Total_Rape_per_Area'].sum()

print("\n \033[1;34;47m Total Rape in entire India -")

print(y)
# # Data to plot

labels = 'Total Forced Rape by Kidnapping', 'Total_Rapes_in_India'

sizes = [x, y]

colors = ['seagreen', 'royalblue']

explode = (0.1, 0)  # explode 1st slice



# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)

plt.tight_layout()

plt.axis('equal')

plt.title("Percentage of Known vs Unknown Offenders in cases of Rape in India",color = 'blue',fontsize=14)

plt.show()
total_crime = pd.read_csv('../input/crime20012012/01_District_wise_crimes_committed_IPC_2001_2012.csv',na_filter = 'False')

total_crime.sample(5).style.set_table_styles(

[{'selector': 'tr:hover',

  'props': [('background-color', 'yellow')]}]

)
total_crime.fillna('')

total_crime.isnull().sum().sum()

total_crime = total_crime[(total_crime['YEAR'] > 2000)  & (total_crime['YEAR']< 2011)]

# considering data only for 2001 to 2010 making it consistent to rest of the EDA.

total_crime.YEAR = pd.to_datetime(total_crime['YEAR'], format ='%Y').dt.strftime('%Y')



""" 

Taking these columns for crime on women - RAPE, KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS, ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY, INSULT TO MODESTY OF WOMEN

These columns are exluded as per below given reasons -

IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES - this links to prostitution/ abuse.

CRUELTY BY HUSBAND OR HIS RELATIVES DOWRY DEATHS - as these are not social crime rather crimes at inter-personal level.

OTHER RAPE column is also excluded as data is not of Female Gender

"""





total_crime.loc[:,'Total_Social_Crime_of_Women'] = total_crime.groupby('YEAR')['RAPE'].transform('sum') + total_crime.groupby('YEAR')['KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS'].transform('sum') + total_crime.groupby('YEAR')['INSULT TO MODESTY OF WOMEN'].transform('sum') + total_crime.groupby('YEAR')['ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY'].transform('sum') 



total_crime.loc[:,'Total_Crime']  = total_crime.groupby('YEAR')['TOTAL IPC CRIMES'].transform('sum')

plot_total_crime  = total_crime.drop_duplicates('YEAR', keep = 'first', inplace = False)

fig, ax = plt.subplots(figsize=(14,10))

X = plot_total_crime['YEAR']

A = plot_total_crime['Total_Social_Crime_of_Women']

B = plot_total_crime['Total_Crime']



plt.bar(X, A, color = 'royalblue', label='Total Social Crime on Women in India engaging her modesty including Rape', edgecolor='black')

plt.bar(X, B, color = 'seagreen', bottom = A, label='Total IPC Crime in India', edgecolor='black')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',  borderaxespad=0.)

plt.title('Proportion of Crime on Women vs total IPC crimes in India ',color = 'blue', fontsize=14)

#plt.tight_layout()

plt.show()
# this dataset has issues, so ignore those bad cells in another column -17.

arrests_crime_women = pd.read_csv('../input/crime-in-india/43_Arrests_under_crime_against_women.csv',error_bad_lines=False,na_filter = 'False', warn_bad_lines=False)

arrests_crime_women.sample(5).style.set_table_styles(

[{'selector': 'tr:hover',

  'props': [('background-color', 'yellow')]}]

)
# data cleaning and understanding it better



arrests_crime_women.fillna('')

arrests_crime_women.isnull().sum().sum()

arrests_crime_women.columns
# change the format - Year

arrests_crime_women.Year = pd.to_datetime(arrests_crime_women['Year'], format ='%Y').dt.strftime('%Y')

action_on_rape_cases = arrests_crime_women[arrests_crime_women['Group_Name'] == 'Rape']



action_on_rape_cases.loc[:,'Total_Persons_Arrested'] = action_on_rape_cases.groupby('Year')['Persons_Arrested'].transform('sum')

action_on_rape_cases.loc[:,'Total_Persons_Convicted'] = action_on_rape_cases.groupby('Year')['Persons_Convicted'].transform('sum')

action_on_rape_cases.loc[:,'Total_Persons_Trial_Completed'] = action_on_rape_cases.groupby('Year')['Persons_Trial_Completed'].transform('sum')



plot_action_on_rape_cases  = action_on_rape_cases.drop_duplicates('Year', keep = 'first', inplace = False)
fig, ax = plt.subplots(figsize=(14,10))

#plot_total_rape.plot(legend=False)

x = total_rape['Year']

y = total_rape['Total_Rape_per_Year']

plt.ylim(0, 30000)

#plt.plot(x,y)

plt.bar(x,y, edgecolor='black')



plt.xlabel('Year')

plt.ylabel('Count')



x=action_on_rape_cases['Year']

Ya = action_on_rape_cases['Total_Persons_Arrested']

plt.plot(x, Ya, color='seagreen', marker='o', linestyle='dashed' , label='Total_Persons_Arrested')

Yb = action_on_rape_cases['Total_Persons_Convicted']

plt.plot(x, Yb, color='yellow', marker='o', linestyle='dashed', label='Total_Persons_Convicted')

Yc = action_on_rape_cases['Total_Persons_Trial_Completed']

plt.plot(x, Yc, color='indigo', marker='o', linestyle='dashed' , label='Total_Persons_Trial_Completed')



plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',  borderaxespad=0.)

plt.title('Number of rapes per year in all of India for the decade (2001 - 2010) and status of action by Police and Courts',color = 'blue', fontsize=14)

#plt.tight_layout()

plt.show()
# 2001 statistics 

population_2001 = pd.read_csv('../input/census2001/all.csv', na_filter = False)

population_2001.head(3)

#population_2001.isnull().sum().sum()

Total_Indian_popu_2001 = population_2001['Persons'].sum()

Total_Indian_popu_2001





# 2011 statistics 

population_2011 = pd.read_csv('../input/india-census/india-districts-census-2011.csv', na_filter = False)

population_2011.head(3)

Total_Indian_popu_2011 = population_2011['Population'].sum()

Total_Indian_popu_2011

Total_Indian_popu_2010 = (Total_Indian_popu_2011 * 0.988).astype('int64')  # decrease by 1.2% and converting it to an int as population can be a whole number only.

#Total_Indian_popu_2010 # estimated population of India in 2010 year
#total_rape in 2001

hundred_k = 100000

rape_2001 = total_rape[total_rape['Year'] == '2001']

total_rape_2001 = rape_2001['Rape_Cases_Reported'].sum()

total_rape_2001



#total_rape in 2005

rape_2005 = total_rape[total_rape['Year'] == '2005']

total_rape_2005 = rape_2005['Rape_Cases_Reported'].sum()

total_rape_2005



#total_rape in 2010

rape_2010 = total_rape[total_rape['Year'] == '2010']

total_rape_2010 = rape_2010['Rape_Cases_Reported'].sum()

total_rape_2010



# Count of Rape per Lakh people in 2001

rape_cnt_2001_per_lk = total_rape_2001/hundred_k



# Count of Rape per Lakh people in 2001

rape_cnt_2005_per_lk = total_rape_2005/hundred_k



# Count of Rape per Lakh people in 2010

rape_cnt_2010_per_lk = total_rape_2010/hundred_k

#print(rape_cnt_2001_per_lk)

#print(rape_cnt_2010_per_lk)



Total_Indian_popu_2005 = (Total_Indian_popu_2001+  Total_Indian_popu_2010)/2 # making assumption that 2005 population is avg of 2001 and 2011



# Percent of Rape as per Indian population in 2001

rape_perc_2001 = total_rape_2001/Total_Indian_popu_2001



# Percent of Rape as per Indian population in 2005

rape_perc_2005 = total_rape_2005/Total_Indian_popu_2005



# Percent of Rape as per Indian population in 2010

rape_perc_2010 = total_rape_2010/Total_Indian_popu_2010

#print(rape_perc_2001)

#print(rape_perc_2010)



data_df = [[2001,rape_cnt_2001_per_lk,rape_perc_2001,rape_perc_2001],[2005,rape_cnt_2005_per_lk,rape_perc_2005,rape_perc_2001*1.07*1.07*1.06*1.06 ], [2010,rape_cnt_2010_per_lk,rape_perc_2010,rape_perc_2001*1.07*1.07*1.06*1.06*1.05*1.05*1.05*1.04*1.04 ]] 

# population increase in 2005 is 1.6%

# population increase in 2010 is 1.4%



plot_df = pd.DataFrame(data_df , columns = ['Year', 'Rape_Count_per_Lk', 'Rape_Percent', 'Population_Trend'])

plot_df.style.set_table_styles(

[{'selector': 'tr:hover',

  'props': [('background-color', 'yellow')]}]

)
#Plotting Rape Rate 

x= plot_df['Year']

y = plot_df['Rape_Count_per_Lk']



fig, ax = plt.subplots(figsize=(14,10))

plt.tight_layout()

ax.set(xlim=(2000, 2011))

s = [200*2**n for n in range(len(x))]

plt.scatter(x,y,s=s)



#plt.xticks(x)

plt.title("Change in Rape Rate in India - 2001 to 2010",color = 'royalblue', fontsize=14)

plt.ylabel('Rape Rate in India', fontsize=14)

plt.xlabel('Years', fontsize=14)



plt.show()
#Plotting Percentage of Rapes as per Population of India

x= plot_df['Rape_Percent']

y= plot_df['Year']

# #z=plot_df['Population_Perc_Incr']

fig, ax = plt.subplots(figsize=(14,10))

#plt.tight_layout()

plt.bar(plot_df['Year'], plot_df['Rape_Percent'].astype(float), color='royalblue', edgecolor='black')

plt.title('Percentage of Rape as per Population in the whole of India in Years - 2001 to 2010 ',color = 'blue', fontsize=14)

plt.xticks(y)

plt.xlabel('Year', fontsize=14)

plt.ylabel('Percentage of Rape', fontsize=14)



plt.show()
x= plot_df['Rape_Percent']

#y= plot_df['Year']

z= plot_df['Population_Trend']

N=3

width= 0.35

f, ax = plt.subplots(figsize=(14,10))

#plt.tight_layout()

plot_data_per= [[plot_df['Rape_Percent'][0], plot_df['Rape_Percent'][1], plot_df['Rape_Percent'][2]],[plot_df['Population_Trend'][0],plot_df['Population_Trend'][1],plot_df['Population_Trend'][2]]]

#plot_data_per_df = df.DataFrame(plot_data_per, columns ='Rape_Percent', 'Year', 'Population_Perc_Incr')

plt.title('Percentage of Rape as per Population increase in the whole of India in Years - 2001 to 2010 ',color = 'blue', fontsize=14)

X= np.arange(3)

ind= np.arange(N)

ax.set_ylabel('Percentage Rise')

ax.set_xticks(ind + width / 2)

ax.set_xticklabels( ('2001', '2005', '2010') )

plt.bar(X + 0.00, plot_data_per[0], color = 'royalblue', width = 0.25, label ='Rape Percent Increase', edgecolor='black')

plt.bar(X + 0.25, plot_data_per[1], color = 'seagreen', width = 0.25, label ='Approx. Population Increase Rate', edgecolor='black')



plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',  borderaxespad=0.)

plt.show()
rape_dfs = pd.read_csv('../input/world-rape-data/inputrape.csv') 

#rape_dfs.sample(5)

#  update names to match names in geoJSON file

rape_dfs.replace(np.nan, 0, inplace=True)

rape_dfs.reset_index(inplace=True)

rape_dfs.replace({

        'United States':'United States of America',

        'Republic of Korea':'South Korea',

        'Russian Federation':'Russia'},

        inplace=True)

rape_dfs.head().style.set_table_styles(

[{'selector': 'tr:hover',

  'props': [('background-color', 'yellow')]}]

)

rape_dfs.isnull().sum().sum()
world_geo = os.path.join('../input/worldcountries', 'world-countries.json')

world_geo
world_choropelth = folium.Map(location=[0, 0], tiles='Mapbox Bright',zoom_start=2)

# 2005 map

world_choropelth.choropleth(

    geo_data=world_geo,

    data=rape_dfs,

    columns=['Country','R2005'],

    key_on='feature.properties.name',

    fill_color='YlGnBu',

    nan_fill_color ='white',

    nan_fill_opacity = 'white',

    fill_opacity=0.7, 

    line_opacity=0.5,

    legend_name='Rape rates per 100k Population - 2005')



folium.LayerControl().add_to(world_choropelth)

# display map

world_choropelth
world_choropelth = folium.Map(location=[0, 0], tiles='Mapbox Bright',zoom_start=2)

# 2010 map

world_choropelth.choropleth(

    geo_data=world_geo,

    data=rape_dfs,

    columns=['Country','R2010'],

    key_on='feature.properties.name',

    fill_color='YlGnBu',

    nan_fill_color ='white',

    nan_fill_opacity = 'white',

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Rape rates per 100k Population - 2010')



folium.LayerControl().add_to(world_choropelth)

# display map

world_choropelth
# # PLOTLY NOT WORKING IN KAGGLE

# map_data = dict(

#         type='choropleth',

#         locations=rape_dfs['Code'],

#         z=rape_dfs['R2005'],

#         text=rape_dfs['Country'],

#         colorscale = 'YlGnBu',

#         colorbar={'title': 'World Rape Rate 2005'}, 

        

#       )



# map_layout = dict(

#     title='World Rape Rate 2005',

#     geo=dict(showframe=False)

# )

# map_actual = go.Figure(data=[map_data], layout=map_layout)

# # fig = dict( data=map_data, layout=map_layout )

# # url = py.plot(fig, filename='d3-world-map')

# iplot(map_actual)

# # PLEASE EXCUSE THIS CODE AS PLOTLY NOT WORKING
# # PLOTLY NOT WORKING IN KAGGLE

# map_data = dict(

#         type='choropleth',

#         locations=rape_dfs['Code'],

#         z=rape_dfs['R2010'],

#         text=rape_dfs['Country'],

#         colorscale = 'YlGnBu',

#         colorbar={'title': 'World Rape Rate 2010'},

#       )



# map_layout = dict(

#     title='World Rape Rate 2010',

#     geo=dict(showframe=False)

# )

# map_actual = go.Figure(data=[map_data], layout=map_layout)

# iplot(map_actual)

# # PLEASE EXCUSE THIS CODE AS PLOTLY NOT WORKING
convict_crime = pd.read_csv('../input/crime-in-india/42_Cases_under_crime_against_women.csv',error_bad_lines=False,na_filter = 'False', warn_bad_lines=False)

convict_crime.isnull().sum().sum()

convict_crime.sample()



convict_crime_women = convict_crime[convict_crime['Group_Name'] == 'Total Crime Against Women']

convict_crime_women.loc[:, 'Total_Cases_Convicted'] = convict_crime_women.groupby('Year')['Cases_Convicted'].transform('sum')

convict_crime_women.loc[:, 'Total_Cases_Trials_Completed'] = convict_crime_women.groupby('Year')['Cases_Trials_Completed'].transform('sum')



convict_crime_women.loc[:,'Convcition_Rate'] = (convict_crime_women['Total_Cases_Convicted'] / convict_crime_women['Total_Cases_Trials_Completed'])*100
sns.set(color_codes=True)

f, ax = plt.subplots(figsize=(14,10))

sns.distplot( convict_crime_women['Convcition_Rate'] ,color="royalblue")

plt.title("Frequency Distribution of Convciton Rate in 2000-2010",color = 'blue',fontsize=14)

crimedata_2016 = pd.read_csv('../input/2016crimedata/Justice_System_Table_3A.7_2016.csv', na_filter = False)

crimedata_2016.isnull().sum().sum()

# Total_Indian_popu_2001 = population_2001['Persons'].sum()

# Total_Indian_popu_2001

crimedata_2016_plot = crimedata_2016[crimedata_2016['Category (Col.2)']== 'Total Crime Against Women']

crimedata_2016_plot.sample().style.set_table_styles(

[{'selector': 'tr:hover',

  'props': [('background-color', 'yellow')]}]

)
fig, ax = plt.subplots(figsize=(14,10))

x = plot_total_rape['Year']

y = plot_total_rape['Total_Rape_per_Year']

#plt.bar(x, height= y)

bars = plt.bar(x, height=y,color = 'royalblue',edgecolor='black') #, width=.4)

xlocs, xlabs = plt.xticks()

xlocs=[i+1 for i in range(0,10)]

xlabs=[i/2 for i in range(0,10)]

# for i, v in enumerate(y):

#     plt.text(xlocs[i] - 0.25, v + 0.01, str(v))

for bar in bars:

    yval = bar.get_height()

    plt.text(bar.get_x(), yval + .005, yval)



plt.title('Rape Count in India - 2001 - 2010',color = 'blue', fontsize=14)

#plt.xticks(xlocs, xlabs)

ax.set_xticklabels( ('2001','2002','2003','2004', '2005','2006','2007', '2008', '2009', '2010') )

plt.show()
rape_victims_2016_use = rape_victims_2016[rape_victims_2016['Category (Col.2)'] == 'Total (All India)']

rape_victims_2016_use.sample().style.set_table_styles(

[{'selector': 'tr:hover',

  'props': [('background-color', 'yellow')]}]

)