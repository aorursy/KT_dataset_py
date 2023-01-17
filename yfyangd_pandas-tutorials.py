import pandas as pd
import numpy as np
list1 = [1,2,3,4]
Series = pd.Series(list1)
Series
Series.values
type(Series)
type(Series.values)
Series2 = pd.Series([1,2,3,4], index=['a','b','c','d'])
Series2
Series2['a']
'a' in Series2
'e' in Series2
Series2['e'] # It will get error message
df = pd.read_csv('../input/titanic/train.csv')
df.head(3)
df = pd.read_csv('../input/titanic/train.csv',index_col=3)
df.head(3)
index = df.index
columns = df.columns
values = df.values
index
columns
values
type(index)
type(columns)
type(values)
values.shape
import sqlite3
import pandas as pd
con = sqlite3.connect("../input/pandastutorials/weather_2012.sqlite")
df = pd.read_sql("SELECT * from weather_2012 LIMIT 3", con)
df
df = pd.read_sql("SELECT * from weather_2012 where temp < 0 order by temp desc ", con)
df.head()
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 3)
plt.rcParams['font.family'] = 'sans-serif'
plt.style.use('ggplot')
df['temp'].plot(figsize=(15, 6),grid=True)
df.describe()
df = pd.read_sql("SELECT * from weather_2012 LIMIT 3", con, index_col='id')
df
df = pd.read_sql("SELECT * from weather_2012 LIMIT 3", con, 
                 index_col=['id', 'date_time'])
df
weather_df = pd.read_csv('../input/pandastutorials/weather_2012.csv')
con = sqlite3.connect("test_db.sqlite")
con.execute("DROP TABLE IF EXISTS weather_2012")
weather_df.to_sql("weather_2012", con)
con = sqlite3.connect("test_db.sqlite")
df = pd.read_sql("SELECT * from weather_2012 ORDER BY Weather LIMIT 3", con)
df
import pyodbc
import pymssql
import cx_Oracle
conn=pyodbc.connect(r'DRIVER={SQL Server Native Client 10.0};SERVER=xxxxxx;DATABASE=xxxxxx;UID=xxxxxxx;PWD=xxxxxxxx')
## for PIP issue, we don't show the database information at here
cursor = conn.cursor()
cursor.execute("exec test")
df = pd.read_csv('../input/titanic/train.csv',index_col=3) ## Assign Stage (col=5) as index
##df.head()
df['Sex'][:3]
df['Sex'][:3].shape
type(df['Sex'][:3])
df[['Sex']][:3]
df[['Sex']][:3].shape
type(df[['Sex']][:3].shape)
##df.head()
df[['Sex','Age','Survived']][:3]
type(df[['Sex','Age','Survived']][:3])
df[['Sex','Age']][:3]
df[['Sex','Age']][:3]
df[['Ag']][:3] ## Run this code will be error
df['Sex','Age'][:3] ## Run this code will be error
df.loc[['Braund, Mr. Owen Harris', 'Heikkinen, Miss. Laina']]
df.loc['Braund, Mr. Owen Harris':'Heikkinen, Miss. Laina']
df.loc[:'Heikkinen, Miss. Laina']
df.loc['Braund, Mr. Owen Harris':'Heikkinen, Miss. Laina':2]
df.loc['Montvila, Rev. Juozas':]
df.loc[['Braund, Mr. Owen Harris', 'Heikkinen, Miss. Laina'], ['Sex','Age','Survived']]
df.loc[['Braund, Mr. Owen Harris', 'Heikkinen, Miss. Laina'], 'Sex']
type(df.loc[['Braund, Mr. Owen Harris', 'Heikkinen, Miss. Laina'], 'Sex'])
df.loc['Braund, Mr. Owen Harris':'Heikkinen, Miss. Laina', ['Sex', 'Age']]
type(df.loc['Braund, Mr. Owen Harris':'Heikkinen, Miss. Laina', ['Sex', 'Age']])
df.loc['Braund, Mr. Owen Harris':'Heikkinen, Miss. Laina']
df.loc[:'Heikkinen, Miss. Laina', 'Age':]
df.loc[:, ['Sex','Age']]
df.loc[['Braund, Mr. Owen Harris','Heikkinen, Miss. Laina'], :]
df.loc[['Braund, Mr. Owen Harris','Heikkinen, Miss. Laina']]
rows = ['Braund, Mr. Owen Harris','Heikkinen, Miss. Laina']
cols = ['Age', 'Sex', 'PassengerId', 'Survived', 'Pclass']
df.loc[rows, cols]
df.iloc[4]
df.iloc[[5, 2, 4]]           # remember, don't do df.iloc[5, 2, 4]  Error!
df.iloc[3:5]
df.iloc[3:]
df.iloc[3::2]
df.iloc[[2,3], [0, 4]]
df.iloc[3:6, [1, 4]]
df.iloc[2:5, 2:5]
df.iloc[0, 2]
type(df.iloc[0, 2])
df.iloc[:, 5]
Age = df['Age']
Age.unique()
Age.loc['Braund, Mr. Owen Harris']
Age.loc[['Braund, Mr. Owen Harris', 'Heikkinen, Miss. Laina']]
Age.loc['Braund, Mr. Owen Harris':'Heikkinen, Miss. Laina']
Age.iloc[0]
Age.iloc[[4, 1, 3]]
Age.iloc[4:6]
some_list = ['a', 'two', 10, 4, 0, 'asdf', 'mgmt', 434, 99]
some_list[5]
some_list[-1]
some_list[:4]
some_list[3:]
some_list[2:6:3]
d = {'a':1, 'b':2, 't':20, 'z':26, 'A':27}
d['a']
d['A']
df2 = pd.read_csv('../input/titanic/train.csv')
df2.head(3)
df2.index
range(7)
list(df2.index)
list(range(7))
df2.loc[[2, 4, 5], ['Age', 'Sex']]
df2.iloc[[2, 4, 5], [3,2]]
df2.iloc[:3]
df2.loc[:3]
df2_idx = df2.set_index('Name')
df2_idx
df[3:6]
df['Allen, Mr. William Henry':'Moran, Mr. James']
df.iloc[3:6]      # More explicit that df[3:6]
df.loc['Allen, Mr. William Henry':'Moran, Mr. James']     # more explicit than df['Aaron':'Christina']
df.head(3)
df3 = df.pivot_table(values='Survived',index=['Sex'],columns=['Age'],aggfunc=np.mean)
df3.head()
df3 = df.pivot_table(values='Survived',index=['Sex'],columns=['Age'],aggfunc=np.sum)
df3.head()
df3 = df.pivot_table(values='Survived',index=['Sex'],columns=['Age'],aggfunc='count')
df3.head()
df[3:6, 'Aaron':'Christina'] ## Run this code will be error
df2.Age.head(3)
df[['Age', 'Sex', 'Survived']].head(3)
import pandas as pd
import numpy as np
test = pd.DataFrame(np.random.randn(6,4),columns=['a','b','c','d'])
test
a = []
for i in range(6):
    a.append(test.iloc[i, 0])
print(a)
a[0]