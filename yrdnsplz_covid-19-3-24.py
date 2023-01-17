import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
import numpy as np
cv = pd.read_csv('../input/covid19_3_24.csv')
cv.index = np.arange(1, len(cv)+1) # Starting with 1 instead 0
cv = cv.fillna(0) # Convert NaN to cero
cv
cv.head(15)
cv.Total_Cases = cv.Total_Cases.apply(pd.to_numeric) # Convert Columns to_numeric to plot
cv.New_Cases = pd.to_numeric(cv.New_Cases) # Convert Columns to_numeric to plot
cv.Total_Death = pd.to_numeric(cv.Total_Death) # Convert Columns to_numeric to plot
cv.New_Deaths = pd.to_numeric(cv.New_Deaths) # Convert Columns to_numeric to plot
cv.Total_Recov = pd.to_numeric(cv.Total_Recov) # Convert Columns to_numeric to plot
cv.Active_Cases = pd.to_numeric(cv.Active_Cases) # Convert Columns to_numeric to plot
cv.Serious = pd.to_numeric(cv.Serious) # Convert Columns to_numeric to plot
cv.Tot_Cases_1M = pd.to_numeric(cv.Tot_Cases_1M) # Convert Columns to_numeric to plot
cv.Tot_Deaths_1M = pd.to_numeric(cv.Tot_Deaths_1M) # Convert Columns to_numeric to plot
cv.groupby("Country")[['Total_Cases', 'Total_Death', 'Total_Recov', 'Active_Cases']].sum().reset_index()
from matplotlib.pyplot import figure

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')



topc = cv.Country.head(10)

toptc = cv.Total_Cases.head(10)

toptd = cv.Total_Death.head(10)



plt.bar(topc, toptc, label='Total Cases')

plt.bar(topc, toptd, label='Total Death')

plt.xlabel('Countries')

plt.ylabel('Persons Infected')

plt.xticks(topc, rotation='vertical', size=10)

plt.grid(True)

plt.title('Top 10 Crovid-19 Cases & Death 03-24')

plt.legend(loc="upper right")



plt.show()