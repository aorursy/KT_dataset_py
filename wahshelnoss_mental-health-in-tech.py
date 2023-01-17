import plotly as py

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline
df = pd.read_csv('../input/mental-health-in-tech-2016/mental-heath-in-tech-2016_20161114.csv')

df.info()
# Inspect the df

df.head()
# Get all tech companies

tech = df[df['Is your employer primarily a tech company/organization?'] > 0] # 883 tech companies
tech['What country do you work in?']
countries = df['What country do you work in?'].sort_values().unique()

countries
zmap = dict(df['What country do you work in?'].value_counts().sort_values())

zmap = dict(sorted(zmap.items(), key=lambda x: x[0].lower()))

zval = list(zmap.values())
# Create data dictionary

data = dict(type='choropleth',

    # sunset color scale

    colorscale='sunsetdark',

    # country names 

    locations=countries,

    # codes correspond to country names

    locationmode='country names',

    # show country name as text on hover  

    text=countries,

    # use number of employees as z 

    z=zval,

    # black line spacing in between states

    marker = dict(line = dict(color='rgb(0,0,0)',width=1)),

    # colorbar title legend 

    colorbar={'title':'Employees in Tech Surveyed'}

)
# layout dictionary

layout = dict(title='Locations of Tech Employees Surveyed', geo={'showframe':False,'projection':{'type':'natural earth'}}, width=1000)
# Create choropleth map

choromap = go.Figure(data = [data],layout = layout)

# plot map

iplot(choromap,validate=False)
# 39% of tech employees state they currently have a mental health disorder 

tech['Do you currently have a mental health disorder?'].value_counts(normalize=True)
# 51% of tech employees have had a mental health disorder in the past

tech['Have you had a mental health disorder in the past?'].value_counts(normalize=True)
# 57% of tech employees have sought treatment for a mental health issue

tech['Have you ever sought treatment for a mental health issue from a mental health professional?'].value_counts(normalize=True)
# 49% tech employees diagnosed with mental health condition

tech['Have you been diagnosed with a mental health condition by a medical professional?'].value_counts(normalize=True)
# 5 Most common mental health conditions in tech

tech['If yes, what condition(s) have you been diagnosed with?'].value_counts(normalize=True).head(5)
# Perception: 86% of tech employees feel that being identidied as having a mental health issue would hurt their career

tech['Do you feel that being identified as a person with a mental health issue would hurt your career?'].value_counts(normalize=True)
# Perceptions: 60% of employees think that being open about a mental health issue may have negative consequences

tech['Do you think that discussing a mental health disorder with your employer would have negative consequences?'].value_counts(normalize=True)
# Reality: In 92% of cases, openness about mental health has ensued no negative consequences

tech['Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?'].value_counts(normalize=True)
# 25% think that discussing physical health issues with employer could have negative consequences

tech['Do you think that discussing a physical health issue with your employer would have negative consequences?'].value_counts(normalize=True)
# 60% think that discussing a mental health issue  with employer could have negative consequences

tech['Do you think that discussing a mental health disorder with your employer would have negative consequences?'].value_counts(normalize=True)
# 31% of tech employees feel that their employer takes mental health as seriously as physical health

tech['Do you feel that your employer takes mental health as seriously as physical health?'].value_counts(normalize=True)
# Simplify gender responses

# Note: Approximative feature engineering here to standardize, do not reflect views on gender identities. 

def binaryGender(genders):

    g = str(genders)

    for gender in g: 

        if gender[0].lower() == 'm':

            return 'male'

        elif gender[0].lower() == 'f':

            return 'female'   

        else: 

            return 'other'

        

tech['gender'] = tech['What is your gender?'].apply(binaryGender)
# 22% Women in tech

tech['gender'].value_counts(normalize=True)
# Women in tech are less inclined to discuss a mental health disorder with their employer than men

plt.figure(figsize=(10,5))

sns.countplot(x='gender',data=tech,hue='Do you think that discussing a mental health disorder with your employer would have negative consequences?')
# WFH employees more likely to have mental health benefits

plt.figure(figsize=(10,5))

sns.countplot(x='Do you work remotely?',data=tech,hue='Does your employer provide mental health benefits as part of healthcare coverage?')
# 13% WFH employees do not have mental health benefits 

tech[tech['Do you work remotely?']=='Always']['Does your employer provide mental health benefits as part of healthcare coverage?'].value_counts(normalize=True)
# 28% WIO employees do not have mental health benefits 

tech[tech['Do you work remotely?']=='Never']['Does your employer provide mental health benefits as part of healthcare coverage?'].value_counts(normalize=True)
# WFH employees less likely to think that discussing mental health issues will lead to negative consequences

plt.figure(figsize=(10,5))

sns.countplot(x='Do you work remotely?',data=tech,hue='Do you think that discussing a mental health disorder with your employer would have negative consequences?')
experienced = tech[tech['What is your age?'] > tech['What is your age?'].median()]

novice = tech[tech['What is your age?'] < tech['What is your age?'].median()]
# 61% older employees in tech think discussing mental health may have negative consequences

experienced['Do you think that discussing a mental health disorder with your employer would have negative consequences?'].value_counts(normalize=True)
# 61% younger employees in tech think discussing mental health may have negative consequences 

novice['Do you think that discussing a mental health disorder with your employer would have negative consequences?'].value_counts(normalize=True)
# Age does not make a statistically significant difference in perceptions 

experienced['Do you think that discussing a mental health disorder with your employer would have negative consequences?'].value_counts(normalize=True) - novice['Do you think that discussing a mental health disorder with your employer would have negative consequences?'].value_counts(normalize=True)
# Split employees into developers and not

def dev(role): 

    if ('Developer' in role):

        return 'Developer'

    else:

        return 'Not Developer'



# Create a column called coder to bissect into developers and non-developers

tech['coder'] = tech['Which of the following best describes your work position?'].apply(dev)

tech.head()
# 48% of non-developers have a mental health disorder

tech[tech['coder']== 'Not Developer']['Do you currently have a mental health disorder?'].value_counts(normalize=True)
# 35% of developers have a mental health disorder

tech[tech['coder']== 'Developer']['Do you currently have a mental health disorder?'].value_counts(normalize=True)
# 62% of non-developers have sought treatment from a mental health professional

tech[tech['coder']== 'Not Developer']['Have you ever sought treatment for a mental health issue from a mental health professional?'].value_counts(normalize=True)
# 55% of developers have sought treatment from a mental health professional

tech[tech['coder']== 'Developer']['Have you ever sought treatment for a mental health issue from a mental health professional?'].value_counts(normalize=True)
tech['If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?'].value_counts(normalize=True)
tech['If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?'].value_counts(normalize=True)
# 33% more likely of mental health issue interfering with work often without effective treatment

tech['If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?'].value_counts(normalize=True) - tech['If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?'].value_counts(normalize=True) 
# get the sizes of various tech companies

tech['How many employees does your company or organization have?'].unique()
# Split df into tech startups and large tech companies based on employee number

startups = tech[(tech['How many employees does your company or organization have?'] == '1-5') | (tech['How many employees does your company or organization have?'] == '6-25')]

bigTech = tech[(tech['How many employees does your company or organization have?'] == '500-1000') | (tech['How many employees does your company or organization have?'] == 'More than 1000')]
len(startups) # 234 tech startups
len(bigTech) # 196 big tech companies
# 68% tech startups do not offer resources to learn about mental health  

startups['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'].value_counts(normalize=True).plot.bar(title='Mental Health Awareness Resources at Tech Startups')
# 24% big tech companies do not offer resources to learn about mental health  

bigTech['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'].value_counts(normalize=True).plot.bar(title='Mental Health Awareness Resources at Big Tech Companies')
# Difference: Big tech companies much more likely to have resources to learn more about mental health

diff = startups['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'].value_counts(normalize=True) - bigTech['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'].value_counts(normalize=True)

diff.plot.bar(title='Mental Health Awareness Resources at Big Tech Companies')
# 29% startup employees comfortable to talk about mental health with coworkers

startups['Would you feel comfortable discussing a mental health disorder with your coworkers?'].value_counts(normalize=True)
# 25% big tech employees comfortable to talk about mental health with coworkers

bigTech['Would you feel comfortable discussing a mental health disorder with your coworkers?'].value_counts(normalize=True)
# 40% employees at startups comfortable to talk about mental health with superviser

startups['Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?'].value_counts(normalize=True).plot.bar()
# 35% big tech company employees comfortable to talk about mental health with superviser

bigTech['Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?'].value_counts(normalize=True).plot.bar()
# Difference: Employees at startups more willing to talk to superviser than big tech employees

diff = startups['Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?'].value_counts(normalize=True) - bigTech['Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?'].value_counts(normalize=True)

diff.plot.bar()
# 25% startups provide mental health services 

startups['Does your employer provide mental health benefits as part of healthcare coverage?'].value_counts(normalize=True).plot.bar(title='Mental Health Services at Tech Startups')
# 60% mental Health Services at Big Tech Companies

bigTech['Does your employer provide mental health benefits as part of healthcare coverage?'].value_counts(normalize=True).plot.bar(title='Mental Health Services at Big Tech Companies')
# Difference: Big tech companies more likely to have mental health services for employees

diff = startups['Does your employer provide mental health benefits as part of healthcare coverage?'].value_counts(normalize=True) - bigTech['Does your employer provide mental health benefits as part of healthcare coverage?'].value_counts(normalize=True)

diff.plot.bar(title='Difference in Mental Health Services at Tech Startups and Big Tech')