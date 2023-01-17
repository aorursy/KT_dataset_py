# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')

df.head()
df.info()
df.shape
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels = False,cbar=True)
skills = df['Key Skills'].to_list()

skills = [str(s) for s in skills]

print(skills)
skills = [s.strip().lower()  for i in skills for s in i.split("|")]

#removing nan elements

skills = [s for s in skills if s!='nan']

len(skills)
print(skills)
from collections import Counter



words = skills
key_skills = pd.DataFrame({'Skills':list(Counter(words).keys()),'Skill Importance':list(Counter(words).values())})

key_skills = key_skills.sort_values(['Skill Importance'],ascending = False)

key_skills = key_skills.reset_index(drop=True)

key_skills.head(20)
from wordcloud import WordCloud



combined = " ".join([w for w in skills])

wordcloud = WordCloud(width = 500, height = 500, 

                background_color ='black', 

                min_font_size = 10).generate(combined)

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()
import matplotlib.pyplot as plt
x = key_skills['Skills'].to_list()[0:10]

y = key_skills['Skill Importance'].to_list()[0:10]

plt.title('Skill Count Distribution')

plt.ylabel('Skill Count')

plt.xlabel('Types of Skills')

plt.bar(x,y)

plt.xticks(rotation=45)

plt.show()
job_title = df['Job Title'].value_counts().nlargest(n=10)

job_title
import plotly

import plotly.express as px
fig = px.pie(job_title, 

       values = job_title.values, 

       names = job_title.index, 

       title="Top 10 Job Titles", 

       color=job_title.values,

       color_discrete_sequence=px.colors.qualitative.Prism)

fig.update_traces(opacity=0.7,

                  marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5)

fig.update_layout(title_x=0.5)

fig.show()
function = df['Functional Area'].dropna().to_list()

function = [str(s) for s in function]

function = [s.strip()  for i in function for s in i.split(",")]

function = [ i.split("-")[0] for i in function ]

functional_areas = pd.DataFrame({'Funtional Areas':list(Counter(function).keys()),'Functional Area Importance':list(Counter(function).values())})

functional_areas = functional_areas.sort_values(['Functional Area Importance'],ascending = False)

functional_areas = functional_areas.reset_index(drop=True)

functional_areas.head(20)
combined = " ".join(w for w in function)

wordcloud = WordCloud(width = 500, height = 500, 

                background_color ='green', 

                min_font_size = 10).generate(combined) 

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
ax = sns.barplot(x=functional_areas['Funtional Areas'].to_list()[0:10],y=functional_areas['Functional Area Importance'].to_list()[0:10])

ax.set(xlabel='Application Areas', ylabel='Application Area Importance')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
