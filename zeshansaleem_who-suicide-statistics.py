# Import necessary liberaries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
matplotlib.style.use('ggplot') 
import os
# load data to pandas dataframe
df = pd.read_csv('../input/who_suicide_statistics.csv')
# Review the data first 15 rows to understand it better
df.head(15)
# Drop rows with NaN values, this step is necessary as there are rows where suicide_no is empty 
# and same row has population number otherwise suicide/population will be biased.
df = df.dropna(axis =0)
df.head(15)
# First lets review the suicides with respect to age groups
df.groupby(by=['age'], as_index=False).sum().plot(x='age', y=['suicides_no', 'population'], kind='bar', secondary_y=['population'])
# Now lets see the suicide with respect to gender
df.groupby(by=['sex'], as_index=False).sum().plot(x='sex', y=['suicides_no', 'population'], kind='bar', secondary_y=['population'])
# Plot the suicides and population with respect to years
df.groupby(by=['year'], as_index=False).sum().plot(x='year', y=['suicides_no', 'population'], kind='line', secondary_y=['population'])
dfyearsum = df.groupby(by=['year'], as_index=False).sum()
# calculating suicides per million population
dfyearsum['suicidesperpopulation'] = dfyearsum['suicides_no']*1000000/dfyearsum['population']

dfyearsum.plot(x='year', y=['suicides_no', 'suicidesperpopulation'], kind='line', secondary_y=['suicidesperpopulation'])
# Lets plot the suicides with respect to countaries
df_countrygroup = df.groupby(by=['country'], as_index=False).sum()
df_countrygroup['suicidespercapita'] = df_countrygroup['suicides_no']*1000000/ df_countrygroup['population']

plt.figure(figsize = (12,8))
plt.subplot(2,2,1)
df_countrygroup.sort_values(by=['suicides_no'], ascending=False).head(10).plot(x='country', y=['suicides_no'], kind='bar', title='TOP 10 country with suicides', ax=plt.gca())

plt.subplot(2,2,2)
df_countrygroup.sort_values(by=['suicidespercapita'], ascending=False).head(10).plot(x='country', y=['suicidespercapita'], kind='bar', title='TOP 10 country with suicides/Populattion', ax=plt.gca())

plt.subplot(2,2,3)
df_countrygroup.sort_values(by=['suicides_no'], ascending=True).head(10).plot(x='country', y=['suicides_no'], kind='bar', title='Bottom 10 country with suicides', ax=plt.gca())

plt.subplot(2,2,4)
df_countrygroup.sort_values(by=['suicidespercapita'], ascending=True).head(10).plot(x='country', y=['suicidespercapita'], kind='bar', title='Bottom 10 country with suicides/Populattion', ax=plt.gca())

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, wspace = 0.2, hspace = 1.2)
# Lets see the yearly trends of Russia and Ukrain
df[df['country']=='Russian Federation'].groupby(by=['year'], as_index=False).sum().plot(x='year', y=['suicides_no', 'population'], kind='line', secondary_y=['population'], title= 'Russia')
df[df['country']=='Ukraine'].groupby(by=['year'], as_index=False).sum().plot(x='year', y=['suicides_no', 'population'], kind='line', secondary_y=['population'], title='Ukrain')