import numpy as np  # useful for many scientific computing in Python

import pandas as pd # primary data structure library
!conda install -c anaconda xlrd --yes
df_can = pd.read_excel('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/Canada.xlsx',

                       sheet_name='Canada by Citizenship',

                       skiprows=range(20),

                       skipfooter=2)



print ('Data read into a pandas dataframe!')
df_can.head()

# tip: You can specify the number of rows you'd like to see as follows: df_can.head(10) 
df_can.tail()
df_can.info()
df_can.columns.values 
df_can.index.values
print(type(df_can.columns))

print(type(df_can.index))
df_can.columns.tolist()

df_can.index.tolist()



print (type(df_can.columns.tolist()))

print (type(df_can.index.tolist()))
# size of dataframe (rows, columns)

df_can.shape  
# in pandas axis=0 represents rows (default) and axis=1 represents columns.

df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

df_can.head(2)
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)

df_can.columns
df_can['Total'] = df_can.sum(axis=1)
df_can.isnull().sum()
df_can.describe()
df_can.Country  # returns a series
df_can[['Country', 1980, 1981, 1982, 1983, 1984, 1985]] # returns a dataframe

# notice that 'Country' is string, and the years are integers. 

# for the sake of consistency, we will convert all column names to string later on.
df_can.set_index('Country', inplace=True)

# tip: The opposite of set is reset. So to reset the index, we can use df_can.reset_index()
df_can.head(3)
# optional: to remove the name of the index

df_can.index.name = None
# 1. the full row data (all columns)

print(df_can.loc['Japan'])



# alternate methods

print(df_can.iloc[87])

print(df_can[df_can.index == 'Japan'].T.squeeze())
# 2. for year 2013

print(df_can.loc['Japan', 2013])



# alternate method

print(df_can.iloc[87, 36]) # year 2013 is the last column, with a positional index of 36
# 3. for years 1980 to 1985

print(df_can.loc['Japan', [1980, 1981, 1982, 1983, 1984, 1984]])

print(df_can.iloc[87, [3, 4, 5, 6, 7, 8]])
df_can.columns = list(map(str, df_can.columns))

[print (type(x)) for x in df_can.columns.values] #<-- uncomment to check type of column headers
# useful for plotting later on

years = list(map(str, range(1980, 2014)))

years
# 1. create the condition boolean series

condition = df_can['Continent'] == 'Asia'

print(condition)
# 2. pass this condition into the dataFrame

df_can[condition]
# we can pass mutliple criteria in the same line. 

# let's filter for AreaNAme = Asia and RegName = Southern Asia



df_can[(df_can['Continent']=='Asia') & (df_can['Region']=='Southern Asia')]



# note: When using 'and' and 'or' operators, pandas requires we use '&' and '|' instead of 'and' and 'or'

# don't forget to enclose the two conditions in parentheses
print('data dimensions:', df_can.shape)

print(df_can.columns)

df_can.head(2)
# we are using the inline backend

%matplotlib inline 



import matplotlib as mpl

import matplotlib.pyplot as plt
print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0
print(plt.style.available)

mpl.style.use(['ggplot']) # optional: for ggplot-like style
haiti = df_can.loc['Haiti', years] # passing in years 1980 - 2013 to exclude the 'total' column

haiti.head()
haiti.plot()
haiti.index = haiti.index.map(int) # let's change the index values of Haiti to type integer for plotting

haiti.plot(kind='line')



plt.title('Immigration from Haiti')

plt.ylabel('Number of immigrants')

plt.xlabel('Years')



plt.show() # need this line to show the updates made to the figure
haiti.plot(kind='line')



plt.title('Immigration from Haiti')

plt.ylabel('Number of Immigrants')

plt.xlabel('Years')



# annotate the 2010 Earthquake. 

# syntax: plt.text(x, y, label)

plt.text(2000, 6000, '2010 Earthquake') # see note below



plt.show() 
df_CI = df_can.loc[['India', 'China'], years]

df_CI.head()
df_CI.plot(kind='line')
df_CI = df_CI.transpose()

df_CI.head()
df_CI.index = df_CI.index.map(int)

df_CI.plot(kind='line')
df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)

df_top5 = df_can.head(5)

df_top5 = df_top5[years].transpose()



df_top5.index = df_top5.index.map(int) 



df_top5.plot(kind='line', figsize=(14, 8))



plt.title('Immigration Trend of Top 5 Countries')

plt.ylabel('Number of Immigrants')

plt.xlabel('Years')

plt.show()
df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)



# get the top 5 entries

df_top5 = df_can.head()



# transpose the dataframe

df_top5 = df_top5[years].transpose() 



df_top5.head()
df_top5.index = df_top5.index.map(int) # let's change the index values of df_top5 to type integer for plotting

df_top5.plot(kind='area', 

             stacked=False,

             figsize=(20, 10), # pass a tuple (x, y) size

             )



plt.title('Immigration Trend of Top 5 Countries')

plt.ylabel('Number of Immigrants')

plt.xlabel('Years')



plt.show()
df_top5.plot(kind='area', 

             alpha=0.25, # 0-1, default value a= 0.5

             stacked=False,

             figsize=(20, 10),

            )



plt.title('Immigration Trend of Top 5 Countries')

plt.ylabel('Number of Immigrants')

plt.xlabel('Years')



plt.show()
# option 2: preferred option with more flexibility

ax = df_top5.plot(kind='area', alpha=0.35, figsize=(20, 10))



ax.set_title('Immigration Trend of Top 5 Countries')

ax.set_ylabel('Number of Immigrants')

ax.set_xlabel('Years')
df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)



# get the least 5 entries

df_least5 = df_can.tail(5)



# transpose the dataframe

df_least5 = df_least5[years].transpose() 



df_least5.head()



df_least5.index = df_least5.index.map(int)

df_least5.plot(kind='area', alpha=0.45, stacked=False, figsize=(20, 10))

plt.title('Immigration trend of least 5 countries')

plt.ylabel('Number of immigrants')

plt.xlabel('Years')



plt.show()
# get the 5 countries with the least contribution

df_least5 = df_can.tail(5)



# transpose the dataframe

df_least5 = df_least5[years].transpose() 

df_least5.head()



df_least5.index = df_least5.index.map(int) # let's change the index values of df_least5 to type integer for plotting



ax = df_least5.plot(kind='area', alpha=0.55, stacked=False, figsize=(20, 10))



ax.set_title('Immigration Trend of 5 Countries with Least Contribution to Immigration')

ax.set_ylabel('Number of Immigrants')

ax.set_xlabel('Years')
# np.histogram returns 2 values

count, bin_edges = np.histogram(df_can['2013'])



print(count) # frequency count

print(bin_edges) # bin ranges, default = 10 bins
df_can['2013'].plot(kind='hist', figsize=(8, 5))



plt.title('Histogram of Immigration from 195 Countries in 2013') # add a title to the histogram

plt.ylabel('Number of Countries') # add y-label

plt.xlabel('Number of Immigrants') # add x-label



plt.show()
# 'bin_edges' is a list of bin intervals

count, bin_edges = np.histogram(df_can['2013'])



df_can['2013'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)



plt.title('Histogram of Immigration from 195 countries in 2013') # add a title to the histogram

plt.ylabel('Number of Countries') # add y-label

plt.xlabel('Number of Immigrants') # add x-label



plt.show()
# let's quickly view the dataset  and then generate histogram.

df_can.loc[['Denmark', 'Norway', 'Sweden'], years].plot.hist()
# transpose dataframe

df_t = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()



# generate histogram

df_t.plot(kind='hist', figsize=(10, 6))



plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')

plt.ylabel('Number of Years')

plt.xlabel('Number of Immigrants')



plt.show()
# let's get the x-tick values

count, bin_edges = np.histogram(df_t, 15)



# un-stacked histogram

df_t.plot(kind ='hist', 

          figsize=(10, 6),

          bins=15,

          alpha=0.6,

          xticks=bin_edges,

          color=['coral', 'darkslateblue', 'mediumseagreen']

         )



plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')

plt.ylabel('Number of Years')

plt.xlabel('Number of Immigrants')



plt.show()
count, bin_edges = np.histogram(df_t, 15)

xmin = bin_edges[0] - 10   #  first bin value is 31.0, adding buffer of 10 for aesthetic purposes 

xmax = bin_edges[-1] + 10  #  last bin value is 308.0, adding buffer of 10 for aesthetic purposes



# stacked Histogram

df_t.plot(kind='hist',

          figsize=(10, 6), 

          bins=15,

          xticks=bin_edges,

          color=['coral', 'darkslateblue', 'mediumseagreen'],

          stacked=True,

          xlim=(xmin, xmax)

         )



plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')

plt.ylabel('Number of Years')

plt.xlabel('Number of Immigrants') 



plt.show()
# create a dataframe of the countries of interest (cof)

df_cof = df_can.loc[['Greece', 'Albania', 'Bulgaria'], years]

# transpose the dataframe

df_cof = df_cof.transpose() 



# let's get the x-tick values

count, bin_edges = np.histogram(df_cof, 15)



# Un-stacked Histogram

df_cof.plot(kind ='hist',

            figsize=(10, 6),

            bins=15,

            alpha=0.35,

            xticks=bin_edges,

            color=['coral', 'darkslateblue', 'mediumseagreen']

            )



plt.title('Histogram of Immigration from Greece, Albania, and Bulgaria from 1980 - 2013')

plt.ylabel('Number of Years')

plt.xlabel('Number of Immigrants')



plt.show()
# step 1: get the data

df_iceland = df_can.loc['Iceland', years]



# step 2: plot data

df_iceland.plot(kind='bar', figsize=(10, 6))



plt.xlabel('Year') # add to x-label to the plot

plt.ylabel('Number of immigrants') # add y-label to the plot

plt.title('Icelandic immigrants to Canada from 1980 to 2013') # add title to the plot



plt.show()
df_iceland.plot(kind='bar', figsize=(10, 6), rot=90) # rotate the bars by 90 degrees



plt.xlabel('Year')

plt.ylabel('Number of Immigrants')

plt.title('Icelandic Immigrants to Canada from 1980 to 2013')



# Annotate arrow

plt.annotate('',                      # s: str. Will leave it blank for no text

             xy=(32, 70),             # place head of the arrow at point (year 2012 , pop 70)

             xytext=(28, 20),         # place base of the arrow at point (year 2008 , pop 20)

             xycoords='data',         # will use the coordinate system of the object being annotated 

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)

            )



plt.show()
df_iceland.plot(kind='bar', figsize=(10, 6), rot=90) 



plt.xlabel('Year')

plt.ylabel('Number of Immigrants')

plt.title('Icelandic Immigrants to Canada from 1980 to 2013')



# Annotate arrow

plt.annotate('',                      # s: str. will leave it blank for no text

             xy=(32, 70),             # place head of the arrow at point (year 2012 , pop 70)

             xytext=(28, 20),         # place base of the arrow at point (year 2008 , pop 20)

             xycoords='data',         # will use the coordinate system of the object being annotated 

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)

            )



# Annotate Text

plt.annotate('2008 - 2011 Financial Crisis', # text to display

             xy=(28, 30),                    # start the text at at point (year 2008 , pop 30)

             rotation=72.5,                  # based on trial and error to match the arrow

             va='bottom',                    # want the text to be vertically 'bottom' aligned

             ha='left',                      # want the text to be horizontally 'left' algned.

            )



plt.show()
# sort dataframe on 'Total' column (descending)

df_can.sort_values(by='Total', ascending=True, inplace=True)



# get top 15 countries

df_top15 = df_can['Total'].tail(15)
# generate plot

df_top15.plot(kind='barh', figsize=(12, 12), color='steelblue')

plt.xlabel('Number of Immigrants')

plt.title('Top 15 Conuntries Contributing to the Immigration to Canada between 1980 - 2013')



# annotate value labels to each country

for index, value in enumerate(df_top15): 

    label = format(int(value), ',') # format int with commas

    

    # place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)

    plt.annotate(label, xy=(value - 47000, index - 0.10), color='white')



plt.show()