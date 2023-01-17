from IPython.display import Image

Image('../input/mydata/1naukari.jpg')
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

sns.set_style("darkgrid")

pd.set_option('display.max_columns', 999)



import os

import matplotlib.pyplot as plt

from wordcloud import WordCloud



import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv")

df = pd.DataFrame(df)

df.head().T
df.columns = [i.lower() for i in df.columns]

print("No of rows : ",df.shape[0])

print("No of cols : ",df.shape[1])
plt.rcParams['figure.figsize'] = (10,6)

df.isnull().sum().plot(kind = "bar")

plt.title("Missing Values Count for Each Column ")

plt.show()
# Let's Drop the row if it has atleast one missing values with in it . 

df.dropna(how = 'any',axis = 0,inplace = True)

print("Number of records of DataFrame After Dropping records with missing values : ",df.shape[0])
# Let's double that whether our data has any missing values with in it 

print("no. records with missing values :  ",df.isnull().sum().sum())
df.columns
# Let's drop unwanted features from the Dataset . 

df.drop(['uniq id','crawl timestamp'],axis = 1,inplace = True)
a = df['job title'].value_counts().head(10).values

fig = px.bar(df, 

             y = df['job title'].value_counts().head(10).values, 

             x = df['job title'].value_counts().head(10).index,

             text = a, color = a,

             title = " Top 10 Posted Jobs Title in Naukari.com ",

             labels = {"x":"Top 10 Posted Jobs Title  ","y":"Number of Jobs Title Posted"},

            

            )

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside',marker_line_color='rgb(105,34,73)')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
a = df['location'].value_counts().head(10).values

fig = px.bar(df, 

             y = df['location'].value_counts().head(10).values, 

             x = df['location'].value_counts().head(10).index,

             text = a,

             color = a,

             title = " Number of Jobs Posted in Each Location ",

             labels = {"x":"Top 10 Locations ","y":"Number of Jobs Posted"}

            )

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside',marker_line_color='rgb(8,48,107)')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
plt.rcParams['figure.figsize'] = (8,8)

a = df['key skills'].value_counts().head(10).values

fig = px.bar(df, 

             x = df['key skills'].value_counts().head(10).values, 

             y = df['key skills'].value_counts().head(10).index,

             text = a,

             color = a,

             title = " Top 10 in Demand Key Skills ",

             labels = {"x":" Key Skills ","y":" "},

             orientation = 'h'

   

         

            )

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside',marker_line_color='rgb(8,48,107)')

#fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
names  = df['job experience required'].value_counts().head(10).index

values = df['job experience required'].value_counts().head(10).values

fig = px.pie(df, values=values, names=names, title='Job availablity with respect to Job experience')

fig.show()
# df['role category'].value_counts().head(20).index.to_list()[:20]
imp_words = df['role category'].value_counts().head(20).index.to_list()[:20]



wordcloud = WordCloud(width = 500, height = 500, 

                background_color ='black', 

                min_font_size = 10).generate(str(imp_words))

plt.figure(figsize = (12,12), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()
df['job title'] = df['job title'].str.strip()

ds = df[df['job title']=="Data Scientist"]
a = ds['location'].value_counts().head(10).values

fig = px.bar(df, 

             y = ds['location'].value_counts().head(10).values, 

             x = ds['location'].value_counts().head(10).index,

             text = a,

             color = a,

             title = " Number of Jobs Posted in Each Location ",

             labels = {"x":"Top 10 Locations ","y":"Number of Jobs Posted"}

            )

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside',marker_line_color='rgb(8,48,107)')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
imp_words = ds['key skills'].value_counts().head(10).index.to_list()[:20]



wordcloud = WordCloud(width = 500, height = 500, 

                background_color ='black', 

                min_font_size = 10).generate(str(imp_words))

plt.figure(figsize = (9,9), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()
names  = ds['industry'].value_counts().head(10).index

values = ds['industry'].value_counts().head(10).values

fig = px.pie(df, values=values, names=names, title='Top 10 industries that needs Data Scientist')

fig.show()
names  = ds['role'].value_counts().head(10).index

values = ds['role'].value_counts().head(10).values

fig = px.pie(df, values=values, names=names, title='Top 10 roles under Data Science ')

fig.show()
a = ds['role category'].value_counts().head(10).values

fig = px.bar(df, 

             y = ds['role category'].value_counts().head(10).values, 

             x = ds['role category'].value_counts().head(10).index,

             text = a,

             color = a,

             title = " Top Role Categories under which Data Scientists are working  ",

             labels = {"x":"Top Role Categories ","y": " "}

            )

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside',marker_line_color='rgb(8,48,107)')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
ds = df[df['job title']=="Data Scientist"]

ds['job experience required'] = ds['job experience required'].str.strip()

#ds['job experience required'].value_counts()
ds.loc[ds['job experience required'].str.contains('2 - 5 yrs',case=False), 'job experience required']='Intermediate Professionals'

ds.loc[ds['job experience required'].str.contains('2 - 6 yrs',case=False),'job experience required']='Intermediate Professionals'

ds.loc[ds['job experience required'].str.contains('2 - 7 yrs',case=False),'job experience required']='Intermediate Professionals'

ds.loc[ds['job experience required'].str.contains('3 - 5 yrs',case=False),'job experience required']='Intermediate Professionals'

ds.loc[ds['job experience required'].str.contains('0 - 5 yrs',case=False),'job experience required']='Freshers'

ds.loc[ds['job experience required'].str.contains('3 - 6 yrs',case=False),'job experience required']='Intermediate Professionals'

ds.loc[ds['job experience required'].str.contains('1 - 3 yrs',case=False),'job experience required']='Freshers'

ds.loc[ds['job experience required'].str.contains('3 - 8 yrs',case=False),'job experience required']='Experienced Professionals'

ds.loc[ds['job experience required'].str.contains('5 - 9 yrs',case=False),'job experience required']='Experienced Professionals'

ds.loc[ds['job experience required'].str.contains('5 - 10 yrs',case=False),'job experience required']='Experienced Professionals'

ds.loc[ds['job experience required'].str.contains('7 - 12 yrs',case=False),'job experience required']='Experienced Professionals'

ds.loc[ds['job experience required'].str.contains('5 - 8 yrs',case=False),'job experience required']='Experienced Professionals'

ds.loc[ds['job experience required'].str.contains('4 - 8 yrs',case=False),'job experience required']='Experienced Professionals'

ds.loc[ds['job experience required'].str.contains('4 - 9 yrs',case=False),'job experience required']='Experienced Professionals'

ds['job experience required'].value_counts()
a = ds['job experience required'].value_counts().values

fig = px.bar(df, 

             y = ds['job experience required'].value_counts().values, 

             x = ds['job experience required'].value_counts().index,

             text = a,

             color = a,

             title = " Experience v/s Data Science Jobs Availablity  ",

             labels = {"x":" Experience Levels  ","y": " "}

            )

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside',marker_line_color='rgb(8,48,107)')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()