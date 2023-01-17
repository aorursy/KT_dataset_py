import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import pyplot as plt

plt.style.use('ggplot')



import seaborn as sns # for making plots with seaborn

color = sns.color_palette()

sns.set(rc={'figure.figsize':(25,15)})



import plotly

plotly.offline.init_notebook_mode(connected=True)

#connected=True means it will download the latest version of plotly javascript library.

import plotly.graph_objs as go



import plotly.figure_factory as ff

import cufflinks as cf





import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv(r'C:\Users\yadav\Jupyter Projects\Playstore App Reviews\googleplaystore.csv')
df.head()
#Total number of apps in the dataset

print('Number of apps in the dataset : ' , len(df))

df.sample(5)
df.shape
df.describe()                         # Summary Statistics
df.info()
df.isnull()
#Count the number of missing values in each column

df.isnull().sum()
#Check how many ratings are more than 5 - Outliers

df[df.Rating > 5]
df.drop([10472],inplace=True)
df[10470:10475]
threshold = len(df)* 0.1

threshold
df.dropna(thresh=threshold, axis=1, inplace=True)

df.isnull().sum()
#Define a function impute_median 

#Fills the null values with median

def impute_median(series):

    return series.fillna(series.median())
df.Rating = df['Rating'].transform(impute_median)
df.isnull().sum()
# modes of categorical values

print(df['Type'].mode())

print(df['Current Ver'].mode())

print(df['Android Ver'].mode())
# Fill the missing categorical values with mode

df['Type'].fillna(str(df['Type'].mode().values[0]), inplace=True)

df['Current Ver'].fillna(str(df['Current Ver'].mode().values[0]), inplace=True)

df['Android Ver'].fillna(str(df['Android Ver'].mode().values[0]),inplace=True)
#Count the number of null values

df.isnull().sum()
# - Installs : Remove + and ,



df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)

df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)

df['Installs'] = df['Installs'].apply(lambda x: int(x))

# - Size : Remove 'M','k',and divide by 10^3 



df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)



df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)

df['Size'] = df['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)

df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)





df['Size'] = df['Size'].apply(lambda x: float(x))

df['Installs'] = df['Installs'].apply(lambda x: float(x))



df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))

df['Price'] = df['Price'].apply(lambda x: float(x))



df['Reviews'] = df['Reviews'].apply(lambda x: int(x))
df.head()
number_of_apps_in_category = df['Category'].value_counts().sort_values(ascending=True)



data = [go.Pie(

        labels = number_of_apps_in_category.index,

        values = number_of_apps_in_category.values,

        hoverinfo = 'label+value'

    

)]



plotly.offline.iplot(data)
grp = df.groupby('Category')

x = grp['Rating'].agg(np.mean)

y = grp['Price'].agg(np.sum)

z = grp['Reviews'].agg(np.mean)

#Grouping apps categorically and then getting mean of a particular type

#of app's rating and total sum of price of a particular category and then plotting them.
plt.figure(figsize=(16,5))

plt.plot(x,'ro', color='g')

plt.xticks(rotation=90)

plt.title('Category wise Rating')

plt.xlabel('Categories-->')

plt.ylabel('Rating-->')

plt.show()
plt.figure(figsize=(16,5))

plt.plot(y,'r--', color='b')

plt.xticks(rotation=90)

plt.title('Category wise Pricing')

plt.xlabel('Categories-->')

plt.ylabel('Prices-->')

plt.show()
plt.figure(figsize=(16,5))

plt.plot(z,'bs', color='g')

plt.xticks(rotation=90)

plt.title('Category wise Reviews')

plt.xlabel('Categories-->')

plt.ylabel('Reviews-->')

plt.show()
data = [go.Histogram(

        x = df.Rating,

        xbins = {'start': 1, 'size': 0.1, 'end' :5}

        

)]

layout = {"title":"Average rating","xaxis":{"title":"abc"}}

print('Average app rating = ', np.mean(df['Rating']))

plotly.offline.iplot(data, filename='overall_rating_distribution')
paid_apps = df[df.Price>0]

p = sns.jointplot( "Price", "Rating", paid_apps)
trace0 = go.Box(

    y=np.log10(df['Installs'][df.Type=='Paid']),

    name = 'Paid',

    marker = dict(

        color = 'rgb(214, 12, 140)',

    )



)

trace1 = go.Box(

    y=np.log10(df['Installs'][df.Type=='Free']),

    name = 'Free',

    marker = dict(

        color = 'rgb(0, 128, 128)',

    )

)

layout = go.Layout(

    title = "Number of downloads of paid apps Vs free apps",

    yaxis= {'title': 'Number of downloads (log-scaled)'}

)

data = [trace0, trace1]

plotly.offline.iplot({'data': data, 'layout': layout})