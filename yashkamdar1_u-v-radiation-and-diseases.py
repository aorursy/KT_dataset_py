import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)  # or 1000

pd.set_option('display.max_rows', None)  # or 1000

pd.set_option('display.max_colwidth', -1)  # or 199

import warnings

warnings.filterwarnings('ignore')



%pylab inline
pd.options.display.max_rows = 999
df1 = pd.read_csv('../input/infectious-diseases-by-county-year-and-sex.csv', index_col=None)

df2 = pd.read_csv('../input/uv-county.csv', index_col=None)

df1.shape
df1.columns
df1.describe()
df1.head()

df1['County'].value_counts().size
df2.columns
df2.shape
df2.describe()
df2.head()
df2['COUNTY NAME'].value_counts().size
df2['COUNTY NAME'] = df2['COUNTY NAME'].str.upper()
df3=df1.merge(df2,left_on='County', right_on='COUNTY NAME')
df3.to_csv("Project_2_merged.csv")
df3.index_col=None
df3.drop(['COUNTY NAME','Lower 95% CI','Upper 95% CI','Rate','COUNTY_FIPS'],axis=1,inplace=True)
df3.isna().sum()
df3.fillna(0,inplace=True)
df3.head()
df3.columns = df3.columns.str.replace('²', '')

df3.rename(columns={'UV_ Wh/m':'Intensity'},inplace=True)

highest_intensity=df3[['County','Intensity','STATENAME']].sort_values(by='Intensity',ascending=False)



highest_intensity.drop_duplicates(subset=['STATENAME'], keep='first', inplace=True)



highest_intensity.head(10)
highest_intensity=df3[['County','Intensity','STATENAME']].sort_values(by='Intensity',ascending=False)

highest_intensity_series=highest_intensity.County.unique()

highest_intensity_series[:10]
df4=df3.pivot_table(values='Cases', index=df3.index, columns='Sex', aggfunc='first')

df4.columns

df4.head()
df4.fillna(0,inplace=True)

df4.head()
df5=pd.concat([df3, df4], axis=1, join='inner')

df5.head()



df5.drop('STATE_ABR',axis=1,inplace=True)



df5.columns=['Disease','County','Year','Sex','Cases','Population','State','Intensity','Female','Male','Total']

df5.head()
df5['County']=df5['County'].str.capitalize()

df5.head()
print("df3: ",df3.shape)

print("df5: ",df5.shape)

print("df4: ",df4.shape)
df_with_filtered=df5[["Disease","County","State","Male","Female","Total","Intensity",'Population']]

df_with_filtered.head()
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

i=0

a=[] #A list to store indexes of rows to be dropped

for index,row in df_with_filtered.iterrows():

    i=i+1    

    if i==3: # On reaching a count of 3, check whether the 3 rows share the Same Disease. The county and state is already ordered. 

        if (df_with_filtered.at[index-2,'Disease']==df_with_filtered.at[index-1,'Disease'] and df_with_filtered.at[index,'Disease']==df_with_filtered.at[index-1,'Disease']):

            df_with_filtered.at[index-2,'Male']=df_with_filtered.at[index-1,'Male']

            df_with_filtered.at[index-2,'Total']=df_with_filtered.at[index,'Total']

            df_with_filtered.at[index-2,'Population']=df_with_filtered.at[index,'Population']

            a.append(index)

            a.append(index-1)

            i=0

    

    
df_with_filtered.drop(index=a,inplace=True)
df_with_filtered.head()

len(df_with_filtered)
df_with_filtered.drop_duplicates(inplace=True)

len(df_with_filtered)
df_with_filtered.isna().sum()
df_with_filtered.drop_duplicates(subset=['Disease', 'County','State','Male','Female','Total','Intensity'], keep="first", inplace=True)
len(df_with_filtered)
df_with_filtered.reset_index(drop = True,inplace=True)

df_with_filtered.reset_index(drop = True).head(10)
highest_intensity_series[:10]
import functools
def conjunction(*conditions):

    return functools.reduce(np.logical_and, conditions)



c_1 = df_with_filtered['County']=="Sierra"

c_2 = df_with_filtered['State']=="New Mexico"





data_filtered_disease_top_10_Sierra= df_with_filtered[['Disease','Total','County','Male']][conjunction(c_1,c_2)].sort_values(by='Total',ascending=False)



data_filtered_disease_top_10_Sierra.drop_duplicates(subset=['Disease'], keep='first', inplace=True)

data_filtered_disease_top_10_Sierra.groupby('Disease').Total.max().sort_values(ascending=False).head()

df_filtered_disease_top_10_Sierra=pd.DataFrame({'Disease':data_filtered_disease_top_10_Sierra.Disease, 'Total_cases':data_filtered_disease_top_10_Sierra.Total}).head()



df_filtered_disease_top_10_Sierra.reset_index(inplace=True)

df_filtered_disease_top_10_Sierra.drop(columns='index')
df_filtered_disease_top_10_Sierra['Disease'].loc[2]= 'Shiga toxin E.coli'

df_filtered_disease_top_10_Sierra
plt.figure(figsize=(16,9))

sns.barplot(x = df_filtered_disease_top_10_Sierra.Disease ,y=df_filtered_disease_top_10_Sierra.Total_cases, data=df_filtered_disease_top_10_Sierra)
df_with_filtered_Population=df5[["Disease","County","State","Male","Female","Total","Population","Intensity"]]

df_with_filtered_Population.head()
df_with_filtered_Population.groupby('Disease').Total.sum().sort_values(ascending=False)

States=df_with_filtered_Population['State'].value_counts()

States=pd.DataFrame({'States':States.index})

States
b=[]

for x in States.States:

    c=df_with_filtered_Population[df_with_filtered_Population['Disease']=='Campylobacteriosis'][df_with_filtered_Population['State']==x].Total.sum()



    b.append(c)

b

    
Intensity=[]

for x in States.States:

    i=df_with_filtered_Population['Intensity'][df_with_filtered_Population['State']==x].mean()

    Intensity.append(i)

Intensity

  
df_with_filtered_Population['Intensity'].mean()
Campylobacteriosis_list = pd.DataFrame(

    {'State': States.States,

     'Total_cases_of_Campylobacteriosis': b,

     'Average_Intensity':Intensity

    })

#Campylobacteriosis_list.reset_index()

#Campylobacteriosis_list.drop(columns='State',inplace=True)

Campylobacteriosis_list.sort_values(by='Total_cases_of_Campylobacteriosis',ascending=False)

Campylobacteriosis_list
Campylobacteriosis_list=Campylobacteriosis_list.head(5)
plt.figure(figsize=(16,9))

ax = Campylobacteriosis_list.plot(x='State', y="Total_cases_of_Campylobacteriosis", kind="line")



Campylobacteriosis_list.plot(x='State', y="Average_Intensity", kind="line", ax=ax, color="C2")

#df.plot(x="X", y="C", kind="bar", ax=ax, color="C3")

matplotlib.rc('figure', figsize=(10,5))

plt.show()
df_distribution_of_Campylobacteriosis_by_state=pd.DataFrame(

    {'State': States.States,

     'Total_cases_of_Campylobacteriosis': b,

    })

#Campylobacteriosis_list.reset_index()

#Campylobacteriosis_list.drop(columns='State',inplace=True)

df_distribution_of_Campylobacteriosis_by_state.sort_values(by='Total_cases_of_Campylobacteriosis',ascending=False)

df_distribution_of_Campylobacteriosis_by_state

df_distribution_of_Campylobacteriosis_by_state.set_index('State',drop=True,inplace=True)

df_distribution_of_Campylobacteriosis_by_state
from matplotlib import pyplot as plt

plt.figure(figsize=(16,12))



matplotlib.pyplot.pie(x=df_distribution_of_Campylobacteriosis_by_state['Total_cases_of_Campylobacteriosis'],labels=df_distribution_of_Campylobacteriosis_by_state.index,autopct='%1.1f%%', startangle=90, rotatelabels=True)

plt.show()


Male_list=[]

for x in States.States:

    each_state_male_cases=df_with_filtered_Population[df_with_filtered_Population['Disease']=='Campylobacteriosis'][df_with_filtered_Population['State']==x].Male.sum()



    Male_list.append(each_state_male_cases)

Male_list

    



Female_list=[]

for x in States.States:

    each_state_female_cases=df_with_filtered_Population[df_with_filtered_Population['Disease']=='Campylobacteriosis'][df_with_filtered_Population['State']==x].Female.sum()



    Female_list.append(each_state_female_cases)

Female_list

    
Total_list=[]

for x in range(len(Male_list)):

    each_state_total_cases=Male_list[x]+Female_list[x]

    Total_list.append(each_state_total_cases)

Total_list

    
df_donut_plot=pd.DataFrame({'States':df_distribution_of_Campylobacteriosis_by_state.index,

                           'Male_cases_with_Campylobacteriosis':Male_list,

                            'Female_cases_with_Campylobacteriosis':Female_list,

                            'Total_cases':Total_list

                           })

df_donut_plot
list_for_each_male_female_pair=[]

for x in range(len(Male_list)):

    each_pair_male=df_donut_plot.loc[x,:].values[1]

    each_pair_female=df_donut_plot.loc[x,:].values[2]

    list_for_each_male_female_pair.append(each_pair_male)

    list_for_each_male_female_pair.append(each_pair_female)





list_for_each_male_female_pair
#Code reference: https://medium.com/@kvnamipara/a-better-visualisation-of-pie-charts-by-matplotlib-935b7667d77f

import matplotlib.pyplot as plt

plt.figure(figsize=(16,9))

# Data to plot

labels = df_donut_plot.States

sizes = df_donut_plot.Total_cases

labels_gender = ['Man','Woman']

sizes_gender = list_for_each_male_female_pair

colors = ['#ff6666', '#ffcc99', '#99ff99', '#66b3ff']

colors_gender = ['#c2c2f0','#ffb3e6']

explode = (0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2) 

explode_gender = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,

                 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,

                 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,

                 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,

                 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,

                 0.1,0.1,0.1,0.1,0.1,0.1)

 

# Plot

plt.pie(sizes, labels=labels, colors=colors, startangle=90,frame=True, explode=explode,radius=3, rotatelabels=True)

plt.pie(sizes_gender,colors=colors_gender,startangle=90, explode=explode_gender,radius=2 )

#Draw circle

centre_circle = plt.Circle((0,0),1.5,color='black', fc='white',linewidth=0)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

 

plt.axis('equal')

plt.tight_layout()



plt.show()








