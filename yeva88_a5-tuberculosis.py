# INTRODUCTION



# I sourced this dataset from Tableau Public because it presented a number of parameters in a way

# that would allow me to practice pivot table type summaries that I frequently perform at work.



# Further, the topic itself is interesting, especially given the years represented in the dataset.

# 1990-2013 time period followed major historical changes in many countries: collapse of the Berlin Wall, 

# collapse of the Soviet Union, massive immigration to the U.S. during the 1990s, etc.



# I wanted to examine population's growth/decline in select countries, as well as how TB rates 

# were behaving globally.
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
import pandas as pd

df = pd.read_csv("../input/TB_Burden_Country-1.csv")
import matplotlib.pyplot as plt
# Previewing data as a whole



df
# Summarizing counts by region



df.groupby("Region").size() 
# Sorting Regional counts above from largest to smallest



tb_by_region = df.groupby("Region").size().sort_values(ascending=False)

tb_by_region
# Calculating the average of the regional counts



regional_mean = tb_by_region.mean()

regional_mean
# Visualizing Regional counts as a horizontal bar chart, sorted from largest to smallest



tb_by_region.plot(kind='barh').invert_yaxis()

plt.axvline(regional_mean, color='black', linestyle='--') # Visualizing the average 
# Another way to visualize Regional counts



df['Region'].value_counts().plot(kind='pie',autopct='%1.1f%%')
# Narrowing the dataset to display only Russia-related records



russia = df.loc[df['Country or territory name'] == 'Russian Federation']

russia
# Before visualizing, checking data type for the Year, Population, and TB columns to ensure we're dealing with numbers



print("Year's data type is:", df['Year'].dtypes)

print("Population's data type is:", df['Estimated total population number'].dtypes)

print("TB's data type is: ", df['Estimated prevalence of TB (all forms) per 100 000 population'].dtypes)
# Visualizing Russia's population between 1990-2013



russia.plot(x='Year',y='Estimated total population number')
# Visualizing tuberculosis in Russia between 1990-2013



russia.plot(x='Year',y='Estimated prevalence of TB (all forms) per 100 000 population')
# Summarizing TB prevalence per region between 1990-2013



tb_per_region_over_time = pd.pivot_table(df, values='Estimated prevalence of TB (all forms) per 100 000 population', index=['Year'], columns='Region',aggfunc=np.sum)

tb_per_region_over_time
# Visualizing TB per region over time to assess trends



tb_per_region_over_time.plot()
tb_europe = tb_per_region_over_time['EUR']

tb_europe
plt.style.use('ggplot')

tb_europe.T.plot(kind='bar')

plt.ylabel('TB in Europe')
# Calculating year-over-year difference in TB in Europe (in numbers)



tb_europe.diff()
# Visualizing year-over-year difference in TB in Europe using a different plot style



plt.style.use('Solarize_Light2')

tb_europe.diff().plot()
# Defining TB value in the Europe dataset for 1990



tb_europe_initial = tb_europe[:1]

tb_europe_initial
# Defining TB value in the Europe dataset for 2013



tb_europe_final = tb_europe[-1:]

tb_europe_final
# I tried to create a pie chart that would take the 2 arguments above 

# (tb_europe_initial and tb_europe_final) to show the difference between the two periods,

# but I couldn't figure it out