# - <a href='#s'>6. Summary/Conclusion</a> 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt # for plotting
import plotly.offline as py
py.init_notebook_mode(connected=True)
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/fd-export.csv")
print("Size of Flaredown data or df is",df.shape)
df.head()
df.info()
df.nunique()                     # number of unique counts in different parameters
df.describe(include=["O"])
df['user_id'] = pd.Categorical(df['user_id'])
df['user_id']= df.user_id.cat.codes
df.head()
df["age"] = df.age.replace(0.0,np.nan)
df.head()
df.age.describe()
df[(df['age'] > 100)].sort_values(by='age',ascending=True).head(10)
df[(df['age'] > 117) | (df['age'] < 0) ].shape  # number of columns to be replaced by NaN
df = df.assign(age = lambda x: x.age.where(x.age.ge(0)))    # ALl negative ages replaced by NaN for consistency
df = df.assign(age = lambda x: x.age.where(x.age.le(118)))  # All ages greater than 117 are replaced by NaN
df[(df['age'] > 117) | (df['age'] < 0) ].shape  # as we can see they are replced
df.age.describe()   # now age statistics makes more sense
print("Total numer of unique users are ",df.user_id.nunique())
df.sex.value_counts()  # Total number of check-ins of differet sex categories
df_sex_unique = pd.DataFrame([{'Number_of_Users' : df[df.sex=="doesnt_say"].user_id.nunique()}
                             ,{'Number_of_Users' : df[df.sex=="other"].user_id.nunique()}
                             ,{'Number_of_Users' :  df[df.sex=="male"].user_id.nunique()}
                             ,{'Number_of_Users' : df[df.sex=="female"].user_id.nunique()}
                             ], index=['Doesnt_say', 'Others', 'Male','Female'])
df_sex_unique.head()
plt.figure(figsize=(10,6))
df_sex_unique.Number_of_Users.plot(kind='pie')
df.trackable_type.value_counts()
df.trackable_type.value_counts().plot(kind='barh')
print("Total numer of unique symptoms ('trackable_name') tracked are",df[df.trackable_type=="Symptom"].trackable_name.nunique())
df[df.trackable_type=="Symptom"].trackable_name.value_counts().head(10)  # Top 10 different symptoms traced
plt.figure(figsize=(15,15))
sector_name = df[df.trackable_type=="Symptom"].trackable_name.value_counts().iloc[0:50]
sns.barplot(sector_name.values, sector_name.index)
for i, v in enumerate(sector_name.values):
    plt.text(0.8,i,v,color='k',fontsize=10)
plt.xticks(rotation='vertical')
plt.xlabel('Number of cases registered')
plt.ylabel('Symptom Name')
plt.title(" Common symptoms registered")
plt.show()
df1 = df.set_index(['user_id', 'age'])
df1.head()
df1[df1.trackable_type == "Symptom"].trackable_name.head()

df1[df1.trackable_type == "Treatment"].head(10)
print("Total numer of unique weather conditions ('trackable_name') are",df[df.trackable_type=="Weather"].trackable_name.nunique())
df[df.trackable_type=="Weather"].trackable_name.value_counts()
df[df.trackable_name=="temperature_min"].head()
# df[df.trackable_name=="pressure"].trackable_value.unique()
s_max = df[df.trackable_name=="temperature_max"].trackable_value
s_min = df[df.trackable_name=="temperature_min"].trackable_value
max_temp = pd.to_numeric(s_max, errors='coerce')
min_temp = pd.to_numeric(s_min, errors='coerce')
max_temp.describe()
print (("Average maximum temperature recorded is") ,max_temp.describe()['mean'] )
print (("Average mimimun temperature recorded is") ,min_temp.describe()['mean'] )
#Pressure description
pd.to_numeric(df[df.trackable_name=="pressure"].trackable_value, errors='coerce').describe()
#Humidity description
pd.to_numeric(df[df.trackable_name=="humidity"].trackable_value, errors='coerce').describe()
# df[df.trackable_name=="precip_intensity"].trackable_value.unique()
#Precipitation Intensity
pd.to_numeric(df[df.trackable_name=="precip_intensity"].trackable_value, errors='coerce').describe()
print("Total numer of unique conditions are",df[df.trackable_type=="Condition"].trackable_name.nunique())
df[df.trackable_type=="Condition"].trackable_name.value_counts().head(10)
# df[df.trackable_type=="Condition"].trackable_name.value_counts().iloc[0:30].plot(kind='bar')
plt.figure(figsize=(15,15))
sector_name = df[df.trackable_type=="Condition"].trackable_name.value_counts().iloc[0:50]
sns.barplot(sector_name.values, sector_name.index)
for i, v in enumerate(sector_name.values):
    plt.text(0.8,i,v,color='k',fontsize=10)
plt.xticks(rotation='vertical')
plt.xlabel('Number of cases registered')
plt.ylabel('Condition Name')
plt.title("Most Common Diseases Conditions registered")
plt.show()
print("Total numer of unique Treatments are",df[df.trackable_type=="Treatment"].trackable_name.nunique())
df[df.trackable_type=="Treatment"].trackable_name.value_counts().head(10)
plt.figure(figsize=(15,15))
sector_name = df[df.trackable_type=="Treatment"].trackable_name.value_counts().iloc[0:50]
sns.barplot(sector_name.values, sector_name.index)
for i, v in enumerate(sector_name.values):
    plt.text(0.8,i,v,color='k',fontsize=10)
plt.xticks(rotation='vertical')
plt.xlabel('Number of cases ')
plt.ylabel('Treatment Provided')
plt.title("Most Common Treatments Provided")
plt.show()
print("Total numer of unique Tags are",df[df.trackable_type=="Tag"].trackable_name.nunique())
df[df.trackable_type=="Tag"].trackable_name.value_counts().head(10)
plt.figure(figsize=(15,15))
name = df[df.trackable_type=="Tag"].trackable_name.value_counts().iloc[0:50]
sns.barplot(name.values, name.index)
for i, v in enumerate(sector_name.values):
    plt.text(0.8,i,v,color='k',fontsize=10)
plt.xticks(rotation='vertical')
plt.xlabel('Number of cases ')
plt.ylabel('Treatment Provided')
plt.title("Most Common Treatments Provided")
plt.show()
from wordcloud import WordCloud

names = df[df.trackable_type=="Tag"].trackable_name.value_counts().iloc[0:500].index
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud)
plt.title("Wordcloud for Common Tags", fontsize=25)
plt.axis("off")
plt.show() 
print("Total numer of unique Foods are",df[df.trackable_type=="Food"].trackable_name.nunique())
df[df.trackable_type=="Food"].trackable_name.value_counts().head(10)
# df[df.trackable_type=="Food"].trackable_name.value_counts().iloc[0:100].index
from wordcloud import WordCloud

names = df[df.trackable_type=="Food"].trackable_name.value_counts().iloc[0:100].index
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Food Taken by maximum people", fontsize=25)
plt.axis("off")
plt.show() 
# <a id='s'>5. Summary</a>
