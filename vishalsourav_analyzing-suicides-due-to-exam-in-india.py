import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
suicide=pd.read_csv('../input/suicides-in-india/Suicides in India 2001-2012.csv')
population=pd.read_csv('../input/indian-population-census-2011/Table_2_PR_Cities_1Lakh_and_Above.csv',header=3,skipfooter=2)
population.head()
population.info()
suicide.head()
suicide.info()
population['State']=population[population.columns[1]]
population['Males']=population[population.columns[6]]
population['Females']=population[population.columns[7]]

population['Males']=population['Males'].str.replace(',', '')
population['Males']=population['Males'].astype(int)
population['Females']=population['Females'].str.replace(',', '')
population['Females']=population['Females'].astype(int)

male_population=population.groupby('State')['Males'].agg({'total_males':np.sum})
male_population.reset_index(inplace=True)
female_population=population.groupby('State')['Females'].agg({'total_females':np.sum})
female_population.reset_index(inplace=True)
population=pd.merge(male_population,female_population,how='inner',on='State')
suicide=suicide.where(suicide['Total']!=0).dropna()
suicide=suicide.where(suicide['Type']=='Failure in Examination').dropna()
suicide=suicide[(suicide['Age_group']=='15-29')]
suicide=suicide[suicide['Year']==2011]
male_suicide=suicide[suicide['Gender']=='Male']
female_suicide=suicide[suicide['Gender']=='Female']
male_suicide=male_suicide[['State','Total']]
female_suicide=female_suicide[['State','Total']]
male_suicide=male_suicide.rename(columns={'Total':'Males'})
female_suicide=female_suicide.rename(columns={'Total':'Females'})
suicide=pd.merge(male_suicide,female_suicide,how='inner',on='State')

suicide.set_index("State",inplace=True)
suicide.rename(index={'A & N Islands':'Andaman & Nicobar Islands','Delhi (Ut)':'Delhi'},inplace=True)
suicide=suicide.reset_index()
suicide['State']=suicide['State'].apply(lambda x: x.upper())
population.head()
suicide.head()
df=pd.merge(suicide,population,how='inner',on='State')
df['% male']=df['Males']/df['total_males']
df['% female']=df['Females']/df['total_females']
plt.figure(figsize=(20,10))
plt.plot(df['% male'],'-o',label='% male suicides in 2011')
plt.plot(df['% female'],'-',label='% female suicides in 2011')
plt.ylabel('% suicides')
plt.xlabel('States')
plt.title('% suicides of students(age group 15-29 year) in different states of India in year 2011 due to Failure in examination.',fontsize=10)
plt.legend()
ob=plt.gca()
df["State"].iloc[0]="A&N ISLAND"
lt3=list(df["State"].values)
ob.tick_params(bottom=False)
plt.xticks(np.arange(0,22),lt3,rotation=90,fontsize=8)
plt.show()
