import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load data on household income
household_income = pd.read_csv('../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding= 'unicode_escape', index_col=["Geographic Area", "City"])

# Preprocess data for invalid values
for i in range(len(household_income['Median Income'])):
    if not str(household_income['Median Income'][i]).replace('.','',1).isdigit():
        household_income['Median Income'][i] = 0
    else:
        household_income['Median Income'][i] = float(household_income['Median Income'][i])
household_income.head
# Get average income per state
average_income_per_state = household_income.groupby("Geographic Area").apply(lambda df: df['Median Income'].mean())
# Load data on poverty rate
poverty = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding= 'unicode_escape', index_col=["Geographic Area", "City"])

# Preprocess data for invalid values
for i in range(len(poverty['poverty_rate'])):
    if not str(poverty['poverty_rate'][i]).replace('.','',1).isdigit():
        poverty['poverty_rate'][i] = 0
    else:
        poverty['poverty_rate'][i] = float(poverty['poverty_rate'][i])
poverty.head
# Graph average income per state
plt.figure(figsize=(20,6))
plt.title("Average Household Income By State")
plt.ylabel("Average Household Income (in Dollars)")
sns.barplot(x=average_income_per_state.index, y=average_income_per_state)

# Graph average poverty rate per state
average_poverty_per_state = poverty.groupby("Geographic Area").apply(lambda df: df['poverty_rate'].mean())

plt.figure(figsize=(20,6))
plt.title("Percentage People Below Poverty Rate By State")
plt.ylabel("People Below Poverty Rate (Percentage)")
sns.barplot(x=average_poverty_per_state.index, y=average_poverty_per_state)
# Load data on police killings
police = pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding= 'unicode_escape')
police.rename(columns={'state': 'Geographic Area', 'city': 'City'}, inplace=True)
unarmed_deaths = police.loc[police.armed == 'unarmed']

# Graph # of people shot by state (normal and unarmed)
average_killed_per_state = police.groupby("Geographic Area").apply(lambda df: df['name'].count())
average_unarmed_per_state = unarmed_deaths.groupby("Geographic Area").apply(lambda df: df['name'].count())
average_unarmed_AfricanAmericans_per_state = unarmed_deaths.loc[police.race == 'B'].groupby("Geographic Area").apply(lambda df: df['name'].count())

plt.figure(figsize=(20,6))
plt.title("# People Killed By Police By State")
plt.ylabel("# People Killed By Police")
sns.barplot(x=average_killed_per_state.index, y=average_killed_per_state)

plt.figure(figsize=(20,6))
plt.title("# Unarmed People Killed By Police By State")
plt.ylabel("# Unarmed People Killed By Police")
sns.barplot(x=average_unarmed_per_state.index, y=average_unarmed_per_state)

plt.figure(figsize=(20,6))
plt.title("# Unarmed African Americans Killed By Police By State")
plt.ylabel("# Unarmed African Americans Killed By Police")
sns.barplot(x=average_unarmed_AfricanAmericans_per_state.index, y=average_unarmed_AfricanAmericans_per_state)
unarmed_deaths_by_race = unarmed_deaths.groupby("race").apply(lambda df: df['name'].count())
plt.title("# Unarmed People Killed By Police By Race")
plt.pie(unarmed_deaths_by_race, labels=unarmed_deaths_by_race.index)
plt.show()
plt.title("# of People Below Poverty Level vs. Unarmed Deaths by Police")
plt.ylabel("# of Unarmed Deaths by Police")
sns.scatterplot(x=average_poverty_per_state, y=average_unarmed_per_state)
plt.title("# of People Below Poverty Level vs. Unarmed African American Deaths by Police")
plt.ylabel("# of Unarmed Deaths by Police")
sns.scatterplot(x=average_poverty_per_state, y=average_unarmed_AfricanAmericans_per_state)