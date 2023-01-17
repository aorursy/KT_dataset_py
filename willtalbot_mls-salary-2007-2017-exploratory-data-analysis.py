# Import the necessary Python libraries

import numpy as np

import pandas as pd 

import glob, os.path

import matplotlib.pyplot as plt

%matplotlib inline

from subprocess import check_output
# Import all 11 csv files and concatenate them into one dataframe

path = r'../input/'

all_csv_files = glob.glob(os.path.join(path, "*.csv"))

dfs = [pd.read_csv(i).assign(Season=os.path.basename(i)) for i in all_csv_files]

df_mls = pd.concat(dfs, ignore_index=True)
# A new column named "Season" will be created and added to the new dataframe

df_mls['Season'] = df_mls['Season'].str.replace('mls-salaries-', '')

df_mls['Season'] = df_mls['Season'].str.replace('.csv', '')
# Observe the dataframe

df_mls.head()
# Check for any missing values

df_mls.count()
# Observe the missing data

null_data = df_mls[df_mls.isnull().any(axis=1)]

null_data
# Create a new Column "Name" that combines first and last name together 

def fullname(x, y):

    if str(x) == "nan":

        return str(y)

    else:

        return str(x) + " " + str(y)
# Apply fullname functon to the dataframe

df_mls['Name'] = np.vectorize(fullname)(df_mls['first_name'], df_mls['last_name'])
# Check to see if the function worked sucessfully

df_mls.head()
# Drop the 'last_name' and 'first_name' columns

df_mls = df_mls.drop(['last_name', 'first_name'], axis = 1)
# Rearrange the order of the columns

df_mls = df_mls[['Season', 'club', 'Name', 'position', 'guaranteed_compensation', 'base_salary']]

df_mls.head()
# Observe the different 'club' names

df_mls['club'].unique()
# Investigating what 'pool' is

df_mls[df_mls['club'] == 'Pool']
df_mls[df_mls['club'] == 'POOL']
# Create a function to assign rows to "Free Agent"

def free_agent(x):

    if str(x) == "nan":

        return str("Free Agent")

    elif str(x) == 'None':

        return str("Free Agent")

    elif str(x) == 'Pool':

        return str("Free Agent")

    elif str(x) == 'POOL':

        return str("Free Agent")

    else:

        return str(x)
# Apply the free_agent function to the dataframe

df_mls['club'] = df_mls['club'].apply(free_agent)
# Create a new dataframe for free agents and observe the new dataframe

df_free_agents = df_mls[df_mls['club'] == 'Free Agent']

df_free_agents.head(10)
# Remove 'Free Agents' from the dataframe  

df_mls.drop(df_mls[df_mls['club'] == 'Free Agent'].index, inplace=True)
# Drop rows with missing base_salary

df_mls = df_mls[pd.notnull(df_mls['base_salary'])]
# Observe the row with missing position

df_mls[df_mls['position'].isnull()] 

# Assign the missing value the correct value

df_mls.loc[873, 'position'] = str('D/M')
# Observe the different positions

df_mls['position'].unique()
# Create a function to correct positions

def position_fixer(x):

    if x == 'D-M':

        return str('D/M')

    if x == 'F-D':

        return str('D/F')

    if x == 'D-F':

        return str('D/F')

    if x == 'F-M':

        return str('M/F')

    if x == 'M-F':

        return str('M/F')

    if x == 'M-D':

        return str('D/M')

    if x == 'M/D':

        return str('D/M')

    if x == 'MF':

        return str('M/F')

    if x == 'F/M':

        return str('M/F')

    else:

        return str(x)
# Apply the position_fixer function to the dataframe

df_mls['position'] = df_mls['position'].apply(position_fixer)
# Observe the new dataframe

df_mls.count()
df_mls.describe()
df_mls.head(10)
# Observe the Average Salary by Season

df_mls.groupby(by = 'Season')['guaranteed_compensation'].mean()
# Observe the Max Salary by Season

df_mls.groupby(by = 'Season')['guaranteed_compensation'].max()
# Observe the Min Salary by Season

df_mls.groupby(by = 'Season')['guaranteed_compensation'].min()
# Observe the Difference between the Max and Min Salary by Season

df_mls.groupby(by = 'Season')['guaranteed_compensation'].max() - df_mls.groupby(by = 'Season')['guaranteed_compensation'].min()
# There are some potential outliers with this data

df_mls.groupby(by = 'Season')['guaranteed_compensation'].median()
# Highest Paid Player per season

df_mls.sort_values('guaranteed_compensation', ascending=False).groupby('Season', as_index=False).first()
# Highest Paid Player per Position 2007-2017

df_mls.sort_values('guaranteed_compensation', ascending=False).groupby('position', as_index=False).first()
df_top17 = df_mls[df_mls['Season'] == '2017']

df_top17.sort_values('guaranteed_compensation', ascending = False).head(15)
# Ten Highest Paid Players 2007 - 2017

df_final = df_mls.drop_duplicates(subset='Name')

df_final.sort_values('guaranteed_compensation', ascending = False).head(10)
# Highest Paid Player Per Club Each Season

df_mls.sort_values('guaranteed_compensation', ascending=False).groupby(['Season', 'club'], as_index=False).first()
# Highest Paid Player Per Club 2007 to 2017 Season

df_mls.sort_values('guaranteed_compensation', ascending=False).groupby('club', as_index=False).first()
df_free_agents['guaranteed_compensation'].mean()
# Highest Paid Free Agent Per Season

df_free_agents.sort_values('guaranteed_compensation', ascending=False).groupby('Season', as_index=False).first()
# Highest Paid Free Agent Per Season

df_free_agents.sort_values('guaranteed_compensation', ascending=False).groupby('Season', as_index=False).first()
salary_totals = df_mls.groupby('Season').sum()

salary_totals.plot(kind = 'bar', title = "Total Salary by Season",figsize=(10,6)).set_ylabel("Salary (USD in millions)");

# The total salary has increased over the years

salary_median = df_mls.groupby('Season').median()

salary_median.plot(kind = 'bar', title = "Median Salary by Season",figsize=(10,6)).set_ylabel("Salary (USD in millions)");
salary_mean = df_mls.groupby('Season').mean()

salary_mean.plot(kind = 'bar', title = "Average Salary by Season",figsize=(10,6)).set_ylabel("Salary (USD in millions)");
salary_totals_by_club = df_mls.groupby('club').sum()

salary_totals_by_club.plot(kind = 'bar', title = "Total Salary by Club",figsize=(10,6)).set_ylabel("Salary (USD in millions)");
df_mls17 = df_mls[df_mls['Season'] == '2017'].sort_values('guaranteed_compensation').groupby('club').sum()

df_mls17.plot(kind = 'bar', title = "2017 Total Salary by Club",figsize=(10,6)).set_ylabel("Salary (USD in millions)");
df_mls_dc = df_mls[df_mls['club'] == "DC"]

df_max_salary_by_season_dc = df_mls_dc.groupby('Season')['guaranteed_compensation'].max()

df_max_salary_by_season_dc.plot(kind = 'bar', title = "Highest Salary by Season (DC United)",figsize=(10,6), color = 'red').set_ylabel("Salary (USD in millions)");
df_mls_la = df_mls[df_mls['club'] == "LA"]

df_max_salary_by_season_la = df_mls_la.groupby('Season')['guaranteed_compensation'].max()

df_max_salary_by_season_la.plot(kind = 'bar', title = "Highest Salary by Season (La Galaxy)",figsize=(10,6), color = 'purple').set_ylabel("Salary (USD in millions)");