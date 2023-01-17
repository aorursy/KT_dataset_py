import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plot

import seaborn as sns



df = pd.read_csv("../input/Sub_Division_IMD_2017.csv")

df.head()
print("Columns:",df.shape[1],"   Rows:",df.shape[0])
print("Data Types\n",df.dtypes)
print("India Subdivisons Unique Values:",df.SUBDIVISION.unique())

print("Total Sub-divisions:",len(df.SUBDIVISION.unique()))
for index, column in enumerate(df.columns):

    if (index == 0):

        continue

    print(column,": Max=",df[column].max(),"  Average:",round(df[column].mean(),2),"  Min:",df[column].min())
fig = plot.figure(figsize=(20, 10))

ax = sns.boxplot(data=df[df.columns[2:14]])

ax.set_title("Rainfall in Months")

plot.xlabel("Months",size='18')

plot.ylabel("Rainfall (mm)",size='18')

plot.show()
fig = plot.figure(figsize=(20, 10))

ax = sns.boxplot(data=df[df.columns[15:19]])

ax.set_title("Rainfall in Comulative Months")

plot.xlabel("Comulative Months",size='18')

plot.ylabel("Rainfall (mm)",size='18')

plot.show()
fig = plot.figure(figsize=(20, 10))

ax = sns.boxplot(data=df.ANNUAL, orient='h')

ax.set_title("Rainfall in Comulative Months")

plot.xlabel("Rainfall (mm)",size='18')

plot.ylabel("Annual",size='18')

plot.show()
fig = plot.figure(figsize =(20,30))

ax = sns.barplot(y=df.SUBDIVISION, x=df.ANNUAL,color="#db2959")

ax.set_title("Anual Rainfall in Subdivisions")

plot.yticks(size="14")

plot.show()
grouped = df.groupby("YEAR")

year_rainfall_mean = grouped["ANNUAL"].agg(np.mean)

year_rainfall_max = grouped["ANNUAL"].agg(np.max)

year_rainfall_min = grouped["ANNUAL"].agg(np.min)



year_rainfall = pd.DataFrame(index= year_rainfall_mean.index)

year_rainfall['Mean'] = year_rainfall_mean.values

year_rainfall['Max'] = year_rainfall_max.values

year_rainfall['Min'] = year_rainfall_min.values



fig = plot.figure(figsize=(20,10))

ax = sns.lineplot(data = year_rainfall[year_rainfall.columns[0:3]])

plot.ylabel("Rainfall (mm)",size = "18")

plot.xlabel("Year",size = "18")

plot.show()
for index,column in enumerate(df.columns):

    if index == 0 or index == 1:

        continue

    else:

        df[column].fillna(df[column].mean(), inplace = True)
months = df.columns[2:14]



fig, ax = plot.subplots(4,3)

fig.subplots_adjust(0,0,3,3)



month_count = 0



for i in range(0,4,1):

    for j in range(0,3,1):

        sns.stripplot(x=df[months[month_count]],ax=ax[i,j])

        month_count +=1

plot.show()