import pandas as pd

df=pd.read_csv(r'/kaggle/input/co2-emission/co2_emission.csv')

df
#Question 1

# Entity and Co2 emission

df1=df[['Entity','Annual CO₂ emissions (tonnes )']]

df1
#Question 2

#average

df2=df1.groupby('Entity').mean()

#minimum

df2.idxmin()
#maximum

df2.idxmax()
#Question 3

#Top 10 countries with max average

df3 = df2.sort_values(by='Annual CO₂ emissions (tonnes )',ascending=False)

df4=df3.head(10)

df4
#Question 4

#Top 10 countries with min average

df3 = df2.sort_values(by='Annual CO₂ emissions (tonnes )',ascending=True)

df4=df3.head(10)

df4
#Question 5



#Name the 10 countries which produced minimum average CO2 after year 2000. 



df5=df[df['Year']>2000]

df51=df5[['Entity','Annual CO₂ emissions (tonnes )']]

df52=df51.groupby('Entity').mean()

df53 = df52.sort_values(by='Annual CO₂ emissions (tonnes )',ascending=True)

df54=df53.head(10)

print(df54)

#Plotting

from matplotlib import pyplot as plt

plt.plot(df54,'c')
#Question 6

#Name the 10 countries which produced maximum average CO2 after year 2000. 

df53 = df52.sort_values(by='Annual CO₂ emissions (tonnes )',ascending=False)

df54=df53.head(10)

print(df54)

#Plotting

plt.plot(df54,'y')
#Question 7

#Plot yearwise Co2 production of the world between 2012-2019.

df7=df[df['Year']>=2012]

df71=df7[['Entity','Year','Annual CO₂ emissions (tonnes )']]

df72=df71[df71['Entity']=='World']

df73=df72.groupby('Year').sum()

plt.plot(df73,'c')
#Question 8

#compare co2 production of top 5 countries(by max co2 emission) over the years by line plot.

df8=df[['Entity','Year','Annual CO₂ emissions (tonnes )']]

df81=df8[df8['Entity']=='World']

df811=df81.groupby('Year').sum()

plt.plot(df811,'c',label='World')

df82=df8[df8['Entity']=='Russia']

df821=df82.groupby('Year').sum()

plt.plot(df821,'b',label='Russia')

df83=df8[df8['Entity']=='United States']

df831=df83.groupby('Year').sum()

plt.plot(df831,'g',label='United States')

df84=df8[df8['Entity']=='EU-28']

df841=df84.groupby('Year').sum()

plt.plot(df841,'r',label='EU-28')

df85=df8[df8['Entity']=='China']

df851=df85.groupby('Year').sum()

plt.plot(df851,'y',label='China')

plt.legend()