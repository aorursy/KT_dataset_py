import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

#import plotly.plotly as py

#plotly.offline doesn't push your charts to the clouds

import plotly.offline as pyo

import plotly.graph_objs as go

pyo.offline.init_notebook_mode()
df = pd.read_csv('../input/weight-height/weight-height.csv')

df.head()
# Data type of each column.

df.dtypes
# Shape of the dataframe.

df.shape
df.describe()
labels = df['Gender'].value_counts().index

values = df['Gender'].value_counts().values



colors = ['#eba796', '#96ebda']



fig = {'data' : [{'type' : 'pie',

                  'name' : "Gender: Pie chart",

                 'labels' : df['Gender'].value_counts().index,

                 'values' : df['Gender'].value_counts().values,

                 'direction' : 'clockwise',

                 'marker' : {'colors' : ['#9cc359', '#e96b5c']}}], 'layout' : {'title' : 'Gender'}}



pyo.iplot(fig)
plt.style.use('ggplot')



# Histogram of the height

df.Height.plot(kind='hist',color='purple',edgecolor='black',figsize=(10,7))

plt.title('Distribution of Height', size=24)

plt.xlabel('Height (inches)', size=18)

plt.ylabel('Frequency', size=18)
# Histogram of the weight

df.Weight.plot(kind='hist',color='purple',edgecolor='black',figsize=(10,7))

plt.title('Distribution of Weight', size=24)

plt.xlabel('Weight (pounds)', size=18)

plt.ylabel('Frequency', size=18);
# Histogram of the height males and females



df[df['Gender']=='Male'].Height.plot(kind='hist',color='blue',edgecolor='black',alpha=0.5,figsize=(10,7))

df[df['Gender']=='Female'].Height.plot(kind='hist',color='magenta',edgecolor='black',alpha=0.5,figsize=(10,7))

plt.legend(labels=['Males','Females'])

plt.title('Distribution of Height', size=24)

plt.xlabel('Height (inches)', size=18)

plt.ylabel('Frequency', size=18);
# Descriptive statistics male

statistics_male = df[df['Gender']=='Male'].describe()

statistics_male.rename(columns=lambda x:x+'_male',inplace=True)



# Descriptive statistics female

statistics_female = df[df['Gender']=='Female'].describe()

statistics_female.rename(columns=lambda x:x+'_female',inplace=True)



# Dataframe that contains statistics for both male and female

statistics = pd.concat([statistics_male,statistics_female], axis=1)

statistics
import numpy as np



# Best fit polynomials.



df_males = df[df['Gender']=='Male']

df_females = df[df['Gender']=='Female']



# Polynomial males.

male_fit = np.polyfit(df_males.Height,df_males.Weight,1)

# array([   5.96177381, -224.49884071])



# Polynomial females.

female_fit = np.polyfit(df_females.Height,df_females.Weight,1)

# array([   5.99404661, -246.01326575])
# Scatter plots and regression lines.



# Males and Females dataframes.



df_males = df[df['Gender']=='Male']

df_females = df[df['Gender']=='Female']



# Scatter plots.

ax1= df_males.plot(kind='scatter', x='Height',y='Weight', color='blue',alpha=0.5, figsize=(10,7))

df_females.plot(kind='scatter', x='Height',y='Weight', color='magenta',alpha=0.5, figsize=(10,7),ax=ax1)



# Regression lines.

plt.plot(df_males.Height,male_fit[0]*df_males.Height+male_fit[1], color='darkblue', linewidth=2)

plt.plot(df_females.Height,female_fit[0]*df_females.Height+female_fit[1], color='deeppink', linewidth=2)



# Regression equations.

plt.text(65,230,'y={:.2f}+{:.2f}*x'.format(male_fit[1],male_fit[0]),color='darkblue',size=12)

plt.text(70,130,'y={:.2f}+{:.2f}*x'.format(female_fit[1],female_fit[0]),color='deeppink',size=12)



# Legend, title and labels.

plt.legend(labels=['Males Regresion Line','Females Regresion Line', 'Males','Females'])

plt.title('Relationship between Height and Weight', size=24)

plt.xlabel('Height (inches)', size=18)

plt.ylabel('Weight (pounds)', size=18);