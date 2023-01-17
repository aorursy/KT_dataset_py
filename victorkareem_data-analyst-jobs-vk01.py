# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv("../input/data-analyst-jobs/DataAnalyst.csv")
df.head()
df.replace(['-1'], [np.nan], inplace=True)
df.replace(['Unknown / Non-Applicable'],np.nan,inplace=True)
df.replace(['None'], [np.nan], inplace=True)
df.replace(['Unknown'],np.nan,inplace=True)
df.drop(columns='Unnamed: 0',inplace=True)
df.shape
df.rename(columns={'Job Title': 'job_title','Job Description': 'job_description','Salary Estimate': 'salary_estimate','Company Name': 'company_name','Type of ownership': 'type_of_own','Easy Apply':'easy_apply'}, inplace=True)
df.shape

#code to separate salary estimates inot lower and upper bounds

new= df.salary_estimate.str.split(' ', n=1, expand=True)
sal_range=new[0].str.split('-',n=1,expand=True)
df["Upper"]=sal_range[0]
df['Lower']=sal_range[1]
df.head(1)
df.Upper.fillna(method='bfill',inplace=True)
df.Lower.fillna(method='ffill',inplace=True)
#turn upper  into int type

df['Lower']=df['Lower'].str.replace('K','000')
df['Lower']=df['Lower'].str.replace('$','')
df['Upper']=df['Upper'].str.replace('K','000')
df['Upper']=df['Upper'].str.replace('$','')
df['Upper'].replace('',0,inplace=True)
df['Lower'].replace('',0,inplace=True)
df['Upper']=df['Upper'].astype(int)
df['Lower']=df['Lower'].astype(int)
df.head()
df['avg_salary']= (df['Upper']+df['Lower'])/2
df.drop(columns='salary_estimate',inplace=True)
df.head()

new2= df['Location'].str.split(',',n=1,expand=True)
df['State']=new2[0]

df.shape
df['Size']=df['Size'].str.replace('+',' to 0')
new2=df['Size'].str.split('employee',expand=True)
size_range=new2[0].str.split('to',n=1,expand=True)
df['lower size']=size_range[0]
df['upper size']=size_range[1]
df.head(2)

df['upper size'].fillna(method='bfill',inplace=True)
df['lower size'].fillna(method='ffill',inplace=True)
df['upper size']=df['upper size'].astype(int)
df['lower size']=df['lower size'].astype(int)

df['avg size']=(df['upper size']+df['lower size'])/2
df.drop(columns='Size',inplace=True)
df.head(2)
df.company_name.unique()
new3=df['company_name'].str.split('\n',n=1,expand=True)
df['Company name']=new3[0]
df.drop(columns='company_name',inplace=True)
df.head(2)
sns.set(style="ticks", color_codes=True)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20,8))
g=sns.stripplot(x='Sector',y='avg_salary',data=df)
g.set_xticklabels(g.get_xticklabels(),rotation=40,ha='right')
plt.title('Correlation between sectors and salaries')
plt.ylabel('Average Salary ($)')
plt.xlabel('Sector')
plt.show()

sns.set(style='ticks',color_codes=True)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(30,6))
g=sns.countplot(x='Sector',data=df)
g.set_xticklabels(g.get_xticklabels(),rotation=90,ha='right')
plt.title('Number of Jobs by Sector')
plt.ylabel('Jobs')
plt.xlabel('Sector')
plt.show()
#sectors by their average size
sns.set(style='ticks',color_codes=True)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(30,6))
g=sns.swarmplot(x='Sector',y='avg size', data=df)
g.set_xticklabels(g.get_xticklabels(),rotation=40,ha='right')
plt.title('Company Size relative to Sector')
plt.ylabel('Average Size')
plt.xlabel('Sector')
plt.show()
my_data= pd.DataFrame(df.job_title.value_counts()).head(20)
my_data.reset_index(inplace=True)
my_data.rename(columns={'index':'Job Title','job_title':'Count'},inplace=True)

sns.set(style='ticks',color_codes=True)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(30,6))
g=sns.barplot(x='Job Title',y='Count',data=my_data)
g.set_xticklabels(g.get_xticklabels(),rotation=40,ha='right')
plt.title('Jobs by title')
plt.xlabel('Job Title')
plt.ylabel('Number of Jobs')
plt.show()
sns.set(style='ticks',color_codes=True)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(30,6))
g=sns.countplot(x='job_title',data=df,order=df.job_title.value_counts().iloc[:20].index)
g.set_xticklabels(g.get_xticklabels(),rotation=40,ha='right')
plt.title('Jobs by title')
plt.xlabel('Job Title')
plt.ylabel('Number of Jobs')
plt.show()
#Revenue

df.Revenue.unique()
df['Revenue'].replace('$2 to $5 billion (USD)','$2billion to $5billion',inplace=True)
df['Revenue'].replace('$10+ billion (USD)','$10billion to 0',inplace=True)
df['Revenue'].replace('$1 to $2 billion (USD)','$1billion to $2billion',inplace=True)
df['Revenue'].replace('Less than $1 million (USD)','0 to $1million',inplace=True)
df['Revenue'].replace('$100 to $500 million (USD)','$100million to $500million',inplace=True)
df['Revenue'].replace('$50 to $100 million (USD)','$50million to $100million',inplace=True)
df['Revenue'].replace('$1 to $5 million (USD)','$1million to $5million',inplace=True)
df['Revenue'].replace('$25 to $50 million (USD)','$25million to $50million',inplace=True)
df['Revenue'].replace('$10 to $25 million (USD)','$10million to $25million',inplace=True)
df['Revenue'].replace('$500 million to $1 billion (USD)','$500million to $1billion',inplace=True)
df['Revenue'].replace('$5 to $10 million (USD)','$5million to $0million',inplace=True)
df['Revenue'].replace('$5 to $10 billion (USD)','$5billion to $10billion',inplace=True)
df.head(15)
new5=df['Revenue'].str.split('to',n=1,expand=True)
df['lower_rev']=new5[0]
df['upper_rev']=new5[1]
df.head(2)
df.lower_rev.fillna(method='bfill',inplace=True)
df.upper_rev.fillna(method='bfill',inplace=True)
df['lower_rev']=df['lower_rev'].str.replace('$','')
df['lower_rev']=df['lower_rev'].str.replace('million','000000')
df['lower_rev']=df['lower_rev'].str.replace('billion','000000000')
df['upper_rev']=df['upper_rev'].str.replace('$','')
df['upper_rev']=df['upper_rev'].str.replace('million','000000')
df['upper_rev']=df['upper_rev'].str.replace('billion','000000000')
df.head(2)
df['lower_rev']=df['lower_rev'].astype(int)
df['upper_rev']=df['upper_rev'].astype(int)
df['lower_rev'].dtype
df['avg_revenue']=(df['lower_rev']+df['upper_rev'])/2
df.head(2)
import plotly.express as px

fig=px.pie(df,values='avg_revenue',names='Sector',title='Average revenues by sector',
           color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()