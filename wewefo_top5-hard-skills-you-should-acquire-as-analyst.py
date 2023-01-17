import pandas as pd

import numpy as np

import re



import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('whitegrid')
data = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv', index_col=0)

data.head()
# Clean Salary columns. Remove '(Glassdoor est.\)'

data['Salary Estimate'] = data['Salary Estimate'].str.replace(' \(Glassdoor est.\)|\$|K', '')





# Split Expected Salary column to MIN and MAX values

data = pd.concat([data.drop('Salary Estimate', axis=1), data['Salary Estimate'].str.split("-", expand=True).rename({0:'Min Expected Salary', 1:'Max Expected Salary'}, axis=1)], axis=1)





# Extract state from `location` column

data['State'] = data['Location'].str[-2:]





# Drop location column

data.drop('Location', axis=1, inplace=True)





# Drop other useless columns

data.drop(['Company Name', 'Headquarters','Type of ownership','Competitors','Easy Apply'], axis=1, inplace=True)





# Convert variable `Company Size` to categorical

data['Company Size'] = pd.Categorical(data['Size'],

               categories=['1 to 50 employees', '51 to 200 employees', '201 to 500 employees', '501 to 1000 employees', '1001 to 5000 employees','5001 to 10000 employees', '10000+ employees'],

               ordered=True)



data.drop('Size', axis=1, inplace=True)





# Drop rows with missing values in `sector` and `founded`

data = data[~data['Sector'].isin(["-1"])]



data = data[~data['Founded'].isin(["-1"])]



# Calculate age for each company

data['Company age'] = 2020 - data['Founded']





# Convert variable `Revenue` to categorical

data['Revenue'] = data['Revenue'].str.replace("\$| \(USD\)", "")



data['Company Revenue'] = pd.Categorical(data['Revenue'],

               categories=['Less than 1 million', '1 to 5 million', '5 to 10 million', '10 to 25 million', '25 to 50 million', '50 to 100 million', '100 to 500 million', '500 million to 1 billion', '1 to 2 billion', '2 to 5 billion', '5 to 10 billion', '10+ billion', 'Unknown / Non-Applicable'],

               ordered=True)



data.drop('Revenue', axis=1, inplace=True)
data.groupby('State')['Job Title'].count().sort_values(0, False).plot.bar(color='k');
from IPython.display import Image

Image("../input/new-business/kauffman-indicators-chart.png", width=1000)
sns.barplot(x='index', y='Sector', data=data['Sector'].value_counts(normalize=True).head(10).reset_index(), palette='gray')

plt.ylabel('Share of total ads')

plt.xlabel("Sectors")

plt.title("Share of total ads by the Sector", fontdict={'size':14, 'weight':'bold'})

plt.xticks(rotation=90);
# Create a dict of skills as keys and search patterns as values

hard_skills_dict = {

    'Python': r"python",

    'R': r"[\b\s/]r[\s,\.]",

    'Excel': r'excel', 

    'Tableau': r'tableau', 

    'SQL': r'sql', 

    'SAS': r'\bsas\b',

    'SPSS': r'\bSPSS\b',

    'VBA': r'\bvba\b',

    'PowerBI': r'power[\s]BI',

    'PowerQuery': r'power[\s]query',

    'SAP': r"\bSAP\b",

    'AWS': r"\bAWS\b",

    'Git': r"\bGit",

    'Dashboard': r"\bDashboard[s]",

    'Spark': r'Spark',

    'Scala': r'Scala',

    'Matlab': r'Matplotlib',

    'C# or C++': r"\bC[#\+\+]", 

    'Java': r'Java',

    'BigQuery': r"Big[\s]Query",

    'Plotly': r'Plotly',

    'Looker': r'Looker',

    'PowerPivot': r'Power[\s]Pivot',

    'Oracle': r'oracle',

    'UNIX': r'unix',

    'Linux': r'linux'

}
hard_skills = {}



# Loop through skills, and count the frequency

for key, search in hard_skills_dict.items():

    hard_skills[key] = data['Job Description'].str.contains(search, flags=re.IGNORECASE).sum()



    

# Build a DataFrame of skills, counts and frequencies.

skills = pd.DataFrame.from_dict(hard_skills, orient='index').reset_index().rename({'index':'skill', 0:'count'}, axis=1).sort_values('count', 0, False)

skills['freq'] = skills['count'] / data.shape[0]
# Plot a barchart of skills

plt.figure(figsize=(20, 6))

sns.barplot(x='skill', y='freq', data=skills, palette='gray')

plt.xticks(rotation=45)

plt.title("How many times was the skill written?", fontdict={'size':14, 'weight':'bold'})

plt.ylabel("")

plt.xlabel("");