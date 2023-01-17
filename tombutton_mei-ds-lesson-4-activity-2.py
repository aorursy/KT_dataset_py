# import pandas
import pandas as pd

# import matplotlib
import matplotlib.pyplot as plt

#import the data and check by viewing the top rows
male_le_data = pd.read_csv('../input/meilds2/male-life-expectancy.csv')
male_le_data.head()
male_le_data.dtypes
male_le_data['North East'].describe()
# create a plot with Years on the x axis and the regions on the y axis 
male_le_data.plot(x='Years',y=['North East',
                               'North West',
                               'Yorkshire and The Humber', 
                               'East Midlands','West Midlands',
                               'East of England',
                               'London',
                               'South East',
                               'South West' ],
                  figsize=(10, 10))
plt.show()
# create a plot with Years on the x axis and the regions on the y axis 
male_le_data.plot(x='Years',y=['North East',
                               'North West',
                               'Yorkshire and The Humber', 
                               'East Midlands','West Midlands',
                               'East of England',
                               'London',
                               'South East',
                               'South West' ],
                  figsize=(10, 10), yticks=(70,72,74,76,78,80))
plt.show()
# create a plot with Years on the x axis and the regions on the y axis 
male_le_data.plot(x='Years',y=['North East',
                               'North West',
                               'Yorkshire and The Humber', 
                               'East Midlands','West Midlands',
                               'East of England',
                               'London',
                               'South East',
                               'South West' ],
                  figsize=(10, 10))
plt.show()
#import the data and check by viewing the top rows
male_le_data = pd.read_csv('../input/meilds2/male-life-expectancy.csv')
male_le_data.head()
# create a plot with Years on the x axis and the regions on the y axis 
male_le_data.plot(x='Years',y=['North East',
                               'North West',
                               'Yorkshire and The Humber', 
                               'East Midlands','West Midlands',
                               'East of England',
                               'London',
                               'South East',
                               'South West' ],
                  figsize=(10, 10))
plt.show()