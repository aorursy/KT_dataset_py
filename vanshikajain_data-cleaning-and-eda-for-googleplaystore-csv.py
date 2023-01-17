import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('../input/googleplaystore.csv')
df.head()
df.shape
df.isnull().sum()
df['Category'].value_counts()
#One odd value of 1.9
df[df['Category']=='1.9']
df['Type'].value_counts()
df[df['Type']=='0']
df['Installs'].value_counts()
#This row has many odd values in many columns such as categories, installs, type and other null values too. It is better to drop this row
df.drop(index=10472,inplace=True)
sns.countplot('Category',data=df,order=df['Category'].value_counts().index)
plt.xticks(rotation= 'vertical')
plt.show()
df.isnull().sum()
df[df['Type'].isnull()]
df['Type'].value_counts()
#Assuming that if price is 0 then the type should be Free. I'd rename the type for this entry to Free
#df[['Price','Type']]
df['Type'][9148]='Free'
df.isnull().sum()
df.dtypes
#converting reviews into integer
df['Reviews']=df['Reviews'].astype('int')
sns.jointplot('Reviews','Rating',data=df)
df['Installs'].value_counts()
#removing the commas and + from installs
df['Installs']=df['Installs'].str.replace('+','')
df['Installs']=df['Installs'].str.replace(',','')
#converting object into integer
df['Installs']=df['Installs'].astype(int)
sns.jointplot('Installs','Rating',data=df)
df['Price$']=df['Price'].str.replace('$','').astype('float')
sns.distplot(df['Price$'],kde=True,hist=False,rug=True)
#Rug plot is analogous to a histogram with zero-width bins, or a one-dimensional scatter plot. 
df['Price$'].hist()
#Few apps are extremely expensive which is kind of odd, will have to look moew into it
#number of categories and genres
print(df['Category'].nunique(), df['Genres'].nunique())
#currently we are keeping both
#df['Size'].str.replace('M','') - Cannot do this because then won't be able to bring uniformity to data
df['SizeM']=df[df['Size'].str.contains('M')]['Size'].str.replace('M','').astype(float)*1000 #changing in kilobytes
df['SizeM'].fillna(df['Size'],inplace=True)
df['Sizek']=df['SizeM'].str.replace('k','')
df['Sizek'].fillna(df['SizeM'],inplace=True)
df['Sizek']=pd.to_numeric(df['Sizek'], errors='coerce') 
#change the size column to numeric and varies with device would change to null values
#Assuming that majorly apps in the same category would lie in similar size (mb) range
df['Sizek']=df.groupby('Category')['Sizek'].transform(lambda x: x.fillna(x.mean())) #fill null values with category wise mean 
df['Sizek'].hist() 
df.isnull().sum()
#Last Updated Format Changed
df['Last Updated']=pd.to_datetime(df['Last Updated'],infer_datetime_format=True)
df.groupby(df['Last Updated'].dt.year)['Android Ver'].size()
df['Android Ver'].value_counts()
#renaming 4.4W and up to cleaner version
df['Android Ver']=df['Android Ver'].str.replace('4.4W and up','4.4 and up')
yrtover=pd.crosstab(index=df['Last Updated'].dt.year, columns=df['Android Ver'])
pd.set_option('max_columns', 100)
yrtover.head(10)
yrtover.plot(kind="barh", figsize=(15,15),stacked=True)
plt.legend(bbox_to_anchor=(1.0,1.0))
df[df['Android Ver'].isnull()]
sns.jointplot('Installs','Rating',data=df)
df['Rating'].hist(bins=20)
df.dropna(inplace=True)
df.isnull().sum()
columns_to_use=['Category','Rating','Reviews','Sizek','Installs','Type','Price$','Content Rating','Genres','Last Updated']
#Currently for EDA these would provide more value to understand the rating of an app
#These features may have a better correlation with rating
#This hypothesis can be confirmed through EDA
new_df=df[columns_to_use]
new_df.head(4)
sns.countplot('Type',data=new_df)
new_df['Content Rating'].value_counts()
sns.countplot('Content Rating',data=new_df)
plt.xticks(rotation= 'vertical')
plt.show()
sns.pairplot(data=new_df,hue='Type')
sns.countplot('Category',data=new_df,order=new_df['Category'].value_counts().index)
plt.xticks(rotation= 'vertical')
plt.show()
top5=new_df[new_df['Category'].isin(['FAMILY','GAME','TOOLS','PRODUCTIVITY','MEDICAL'])]
sns.barplot('Category','Installs',hue='Type',data=top5)
sns.barplot('Category','Installs',data=new_df,order=new_df.groupby('Category')['Installs'].mean().sort_values(ascending=False).index)
plt.xticks(rotation= 'vertical')
plt.show()
pd.pivot_table(data=new_df, index='Category',values='Installs',aggfunc=np.sum,).sort_values(by='Installs',ascending=False)
#Finally Looking at the correlation
sns.heatmap(new_df.corr(),annot=True,fmt='0.2f')
sns.lmplot('Installs','Reviews',data=new_df,x_jitter=True)
#Though installs is a numerical variable it seems more appropriate to plot it as categorical
sns.boxplot('Installs','Reviews',data=new_df)
plt.xticks(rotation='vertical')
plt.show()
