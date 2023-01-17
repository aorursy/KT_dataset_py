import pandas as pd
df=pd.read_csv(r"../input/data-analyst-jobs/DataAnalyst.csv")

df.head()
df[df['Job Title'] =='Data Analyst'] 
df['Rating'].head
df ['Rating'].min()
df['Rating'].max()
df['Rating'].mean()
df['Rating'].value_counts() 
df.loc[0:4,'Rating']
df.iloc[0:4,0:8]
df[df['Rating']==df['Rating'].max()]['Size']
df[df['Rating'] ==df['Rating'].min()]['Size']
df.sort_values(by=['Job Title','Size'],ascending=[True,False]).head()
df.sort_values(by=['Job Title','Size'],ascending=[True,False]) 
df.sort_values(by='Rating',ascending =True).head()
df.head()
df.groupby(by='Job Title')['Size'].describe() 
pd.crosstab(df['Job Title'],df['Rating'],normalize=True) 
d={'1960':2007,'1999':1990} 

df['Founded']=df['Founded'].map(d)

df.head()

o={1960:'2007',1999:'1990'}

df=df.replace({'Founded':o})

df.head()               
df.pivot_table(['Rating','Size'],['Job Title'],aggfunc='mean') 