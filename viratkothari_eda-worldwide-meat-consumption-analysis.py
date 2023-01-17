# !pip install plotly # ==4.2.1
# Importing libraries



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib.pyplot as plt

%matplotlib inline 



import seaborn as sns

import plotly.express as px

from IPython.display import HTML



import cufflinks as cf

cf.go_offline(connected=None)

import plotly.express as px
# # Mounting Google drive to access the file



# from google.colab import drive

# drive.mount('/content/drive', force_remount=True)



# print("Google Drive Mounted Succssfully!")
# Reading the csv file and creating dataframe



df=pd.read_csv("../input/meatconsumption/meat_consumption_worldwide.csv")

# # Unmounting Google drive



# drive.flush_and_unmount()

# print("Google Drive Unmounted Succssfully!")
# printing sample data



df.head()
# size of dataset



df.shape
# information about columns, datatype etc.



df.info()
# Select duplicate rows except first occurrence based on all columns



duplicateRowsDf = df[df.duplicated()]

 

print("Duplicate Rows except first occurrence based on all columns are :")

print(len(duplicateRowsDf))
# Satistical analysis of Numerical columns



df.describe()
# Analysis of Non-numerical columns



df.describe(include=['O'])
# checking for number of null records



df.isnull().sum()
# Number of unique value for each column



df.nunique()
# Number of unique countries



print(df['LOCATION'].unique())

print("Number of unique countries: %s" % (df['LOCATION'].nunique()))
# type of meat



print(df['SUBJECT'].unique())

print("Type of meat: %s" % (df['SUBJECT'].nunique()))
sns.distplot(df['Value'],kde=False, bins=None)

plt.title('Distribution of Values of meat consumption')

plt.show()
sns.distplot(np.log1p(df['Value']))

plt.show()
dfx = pd.get_dummies(df,columns=['MEASURE'])

dfx
sns.heatmap(dfx.corr(),annot=True)

plt.show()
# Total meat consumption meat type-wise



import random

import matplotlib.colors as mcolors



by_c = df.groupby('SUBJECT')[['Value']].sum().reset_index().sort_values('Value',ascending=False)



labels = by_c["SUBJECT"]

sections = by_c["Value"]

colors = None # random.choices(list(mcolors.CSS4_COLORS.values()),k = 4) # This is to generate random colours



plt.pie(sections, labels=labels,

        startangle=90,

        explode = (0.1, 0.1, 0.1, 0.1),

        autopct = '%1.2f%%',

        # shadow=True,

        radius=2, # size of the pie chart

        colors=colors,

        # wedgeprops = {'linewidth': 1},

        rotatelabels = False)



plt.axis('equal') # Try commenting this out.

plt.title('Total meat consumption meat type-wise')

plt.show()





# Laction-wise Total Meat Consumption



dfx = df.groupby('LOCATION')[['Value']].sum().reset_index().sort_values('Value',ascending=False)



fig = px.bar(dfx,dfx['LOCATION'],dfx['Value'], hover_name='LOCATION',hover_data=['LOCATION'],color='LOCATION',title='Laction-wise Total Meat Consumption.')

# HTML(fig.to_html()) # for colab

fig.show() # generally
# Laction-wise Total Meat Consumption



by_c = df.groupby('SUBJECT')[['Value']].sum().reset_index().sort_values('Value',ascending=False)



fig = px.bar(by_c,by_c['SUBJECT'],by_c['Value'], hover_name='SUBJECT',hover_data=['SUBJECT'],color='SUBJECT',title='Meat type-wise Total Meat Consumption.')

# HTML(fig.to_html()) # for colab

fig.show() # generally
# Meat Consumption Change through the Years



fig = px.scatter(df, x="TIME", y="Value", hover_name='LOCATION',hover_data=['MEASURE'],color='SUBJECT',title='Meat Consumption Change through the Years')

# HTML(fig.to_html()) # for colab

fig.show() # generally
df2=df[df['LOCATION'].isin(['WLD','BRICS','OECD','EU28'])==False]



fig = px.scatter(df2, x="TIME", y="Value",symbol='SUBJECT',hover_data=['MEASURE'],color='LOCATION',hover_name='SUBJECT',

                 title='Meat Production by Country and Type')

# HTML(fig.to_html()) # for colab

fig.show() # for general
# change of meat eating habit



dfMEH = df.loc[df['MEASURE'] == 'THND_TONNE']



dfMEH91To95 = dfMEH.loc[(dfMEH['TIME'] >= 1991) & (dfMEH['TIME'] <= 1995)]

dfMEH96To05 = dfMEH.loc[(dfMEH['TIME'] >= 1996) & (dfMEH['TIME'] <= 2005)]

dfMEH06To10 = dfMEH.loc[(dfMEH['TIME'] >= 2006) & (dfMEH['TIME'] <= 2010)]

dfMEH11To15 = dfMEH.loc[(dfMEH['TIME'] >= 2011) & (dfMEH['TIME'] <= 2015)]

dfMEH21To25 = dfMEH.loc[(dfMEH['TIME'] >= 2021) & (dfMEH['TIME'] <= 2025)]



dfMEH91To95 = dfMEH91To95.groupby(by = ['SUBJECT']).Value.sum()

dfMEH96To05 = dfMEH96To05.groupby(by = ['SUBJECT']).Value.sum()

dfMEH06To10 = dfMEH06To10.groupby(by = ['SUBJECT']).Value.sum()

dfMEH11To15 = dfMEH11To15.groupby(by = ['SUBJECT']).Value.sum()

dfMEH21To25 = dfMEH21To25.groupby(by = ['SUBJECT']).Value.sum()



fig = plt.figure (figsize=(18,7))

fig.suptitle('Change of meat consumption habit over 35 years', size = 22)



ax5 = plt.subplot(1, 5, 1)

ax5.set_title('From 1991 to 1995')

dfMEH91To95.plot.pie(autopct='%1.0f%%')

plt.ylabel("")



ax5 = plt.subplot(1, 5, 2)

ax5.set_title('From 1996 to 2005')

dfMEH96To05.plot.pie(autopct='%1.0f%%')

plt.ylabel("")



ax6 = plt.subplot(1, 5, 3)

ax6.set_title('From 2006 to 2010')

dfMEH06To10.plot.pie(autopct='%1.0f%%')

plt.ylabel("")



ax7 = plt.subplot(1, 5, 4)

ax7.set_title('From 2011 to 2015')

dfMEH11To15.plot.pie(autopct='%1.0f%%')

plt.ylabel("")



ax8 = plt.subplot(1, 5, 5)

ax8.set_title('From 2021 to 2025')

dfMEH21To25.plot.pie(autopct='%1.0f%%')

plt.ylabel("")
# Heatmap



dfx=df.pivot_table(index='TIME',columns='SUBJECT',values='Value',aggfunc = sum)

fig, ax = plt.subplots(figsize=(15,10))         # Sample figsize in inches

sns.heatmap(dfx, annot=True, linewidths=.5, ax=ax)

print("Notebook completed.")