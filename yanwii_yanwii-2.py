import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from dateutil.parser import parse
from matplotlib.pyplot import savefig
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")



train = pd.read_csv('../input/train.csv')
ti = DataFrame(train.Dates)
print(train.tail())
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1)



#make dates like H.M
hourm = [float(i.split(" ")[1].split(":")[0]+"."+i.split(" ")[1].split(":")[1]) for i in list(ti['Dates'])]
ti.index = hourm
hourm = DataFrame(hourm)
hourm.columns = ['hm']
crime_time_con = hourm.hm.value_counts()
x = crime_time_con.index
y = crime_time_con
ax.scatter(y,x,color='#34ABD8')
ax.set_xlabel('Number of crime')
ax.set_ylabel("Time of crime")

year = DataFrame([ j.split("-")[0] for j in list(ti['Dates'])])
year_con = DataFrame(year[0].value_counts().sort_index())
year_con.index.name = 'year'
year_con.columns = ['Number of crime']
year_con.plot(kind="barh",figsize=(8,6),color='#34ABD8')

month = DataFrame([ k.split("-")[1] for k in list(ti['Dates'])])
month_con = DataFrame(month[0].value_counts().sort_index())
month_con.index.name = 'Month'
month_con.columns=['Number of crime']
month_con.plot(kind="bar",figsize=(8,6),color='#34ABD8',rot=30)
week = DataFrame(train.DayOfWeek.value_counts())
week.index.name = 'Week'
week.columns = ['Number of crime']
week.plot(kind='bar',color='#34ABD8',figsize=(8,6),rot=30)
category = DataFrame(train.Category.value_counts())
category.index.name = 'Categry'
category.columns = ['Number of crime']
category.plot(kind='bar',color='#34ABD8',figsize=(8,6),rot=90)
year_category = DataFrame(train.Category)
year_category.index = list(year[0])
year_category = year_category.reset_index()

year_category_con = year_category.groupby(['Category','index']).size().unstack()
year_category_con.plot(kind='bar',subplots=True,figsize=(15,40))


