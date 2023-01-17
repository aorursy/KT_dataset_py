import pandas as pd

import seaborn as sns

import numpy as np

from matplotlib import pyplot as plt

from matplotlib.colors import ListedColormap



%matplotlib inline
df = pd.read_csv("../input/WHO.csv")

pd.set_option('display.max_rows', 500) #Default Maximum rows to be displayed is set to 500

pd.set_option('display.max_columns',500) #Default Maximum columns to be displayed is set to 500

df.head()
df.shape
df.columns
df.set_index('Country',drop=True, inplace=True)

df.head()
df.columns = df.columns.str.replace("&lt;", "<")

df.columns = df.columns.str.replace("&gt;", ">")

df.head()
average_population=df['Population_total'].mean()

Adult_literacy_rate_top_25=df[df['Population_total']>average_population].sort_values(by='Adult literacy rate (%)',ascending=False).head(25)

series_Adult_literacy_rate_top_25=Adult_literacy_rate_top_25['Adult literacy rate (%)']

series_Adult_literacy_rate_top_25
Patents_granted=df['Patents_granted']

Patents_granted_top_25=Patents_granted.sort_values(ascending=False).head(25)

print("Top 25 countries with highest number of Patents granted")

Patents_granted_top_25
series_literacyrate_patents = pd.merge(series_Adult_literacy_rate_top_25,Patents_granted_top_25, how="inner", on="Country")

series_literacyrate_patents.dropna() #Dropped rows with NaN values for either of the two columns
#Average of deaths by liver cancer

Average_of_per_capita_alcohol_consumption=df['Per capita recorded alcohol consumption (litres of pure alcohol) among adults (>=15 years)'].mean()

Average_of_per_capita_alcohol_consumption

print("Average  per capita Alcohol Consumption is: ", Average_of_per_capita_alcohol_consumption, "litres")


df['Deaths by Liver Cancer']=df.Liver_cancer_number_of_male_deaths + df.Liver_cancer_number_of_female_deaths # Concat columns to get total deaths by liver cancer

df['Deaths by Liver Cancer']=df['Deaths by Liver Cancer'].fillna(0) #filling all nana values with 0

df_liver_cancer_analysis=df[['Per capita recorded alcohol consumption (litres of pure alcohol) among adults (>=15 years)','Deaths by Liver Cancer']].sort_values(by='Deaths by Liver Cancer',ascending=False).head(10)

df_liver_cancer_analysis



plt.figure(figsize=(16,9))

# figure ration 16:9

sns.set() # for style

sns.barplot(y=df_liver_cancer_analysis['Deaths by Liver Cancer'],x=df_liver_cancer_analysis['Per capita recorded alcohol consumption (litres of pure alcohol) among adults (>=15 years)'])
df_female_literacy_analysis=df[['Literacy_rate_youth_female','Literacy_rate_adult_female']].sort_values(by="Literacy_rate_adult_female",ascending=False).head(10)

df_female_literacy_analysis
plt.figure(figsize=(16,9))

plt.scatter(x=df_female_literacy_analysis.index,y='Literacy_rate_youth_female', data=df_female_literacy_analysis)

plt.scatter(x=df_female_literacy_analysis.index,y='Literacy_rate_adult_female', data=df_female_literacy_analysis)

plt.legend(['Youth','Adult'], title="Female Literacy Rates")

plt.show()