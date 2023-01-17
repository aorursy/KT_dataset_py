# Import data analysis tools

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#Read and probe data

state = pd.read_csv('../input/statesfinancials/states - states.csv')

state.head()
messyI = state[state['STATE'].str.contains("Illinois")]

messyI
# State Revenue by Year

plt.figure(figsize=(10,5))

plt.title('State Revenue by Year')

plt.scatter(messyI['YEAR'], messyI['STATE_REVENUE'], alpha=0.5)

plt.show()
#Illinois ordered by years

messyI.sort_values(by='INSTRUCTION_EXPENDITURE', ascending=False)
# Instruction Expenditure by year

plt.figure(figsize=(10,5))

plt.title('Instruction Expenditure by Year')

plt.scatter(messyI['YEAR'], messyI['INSTRUCTION_EXPENDITURE'], alpha=0.5)

plt.show()

#Enroll Possibility of people really leaving the state

messyI.sort_values(by='ENROLL', ascending=False)
# State enrollment by year

plt.figure(figsize=(10,5))

plt.title('State Enrollment by Year')

plt.scatter(messyI['YEAR'], messyI['ENROLL'], alpha=0.5)

plt.show()
messyI['Total Expenditure per Student'] = messyI['TOTAL_EXPENDITURE']/messyI['ENROLL']

messyI.sort_values(by='Total Expenditure per Student', ascending=False)
# Total expenditure per student 

plt.figure(figsize=(10,5))

plt.title('Total expenditure per student')

plt.scatter(messyI['YEAR'], messyI['Total Expenditure per Student'], alpha=0.5)

plt.show()
#Playing around with miscellaneous columns. 

# Total revenue has been going up but federal and state varies. For state ever since 2010 revenues have been going up. 

# For total expenditure ever since 2012 that has been consistently rising. 

# Could be a way to attribute to strikes and leadership changes. 

messyI.sort_values(by='TOTAL_EXPENDITURE', ascending=False)