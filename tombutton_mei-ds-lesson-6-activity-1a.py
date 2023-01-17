# import pandas
import pandas as pd 

# import matplotlib
import matplotlib.pyplot as plt
hurn_2015_data = pd.read_csv("../input/ldsedexcel/hurn-2015.csv")
hurn_2015_data.head()
hurn_1987_data = pd.read_csv('../input/ldsedexcel/hurn-1987.csv')
hurn_1987_data.head()
# create a temporary array of the new data
# hurn_1987_data['Date'].str[:5] extracts the first 5 characters of the string for the date
new_data = {'shortdate': hurn_1987_data['Date'].str[:5], 'hurntemp1987' : hurn_1987_data['Daily Mean Temperature']}

# create the dataframe with these columns and the data
all_stations_data = pd.DataFrame (new_data, columns = ['shortdate','hurntemp1987'])

# diplay the top rows to check it has worked
all_stations_data.head()
# add a new column for the 2015 Hurn data 
all_stations_data['hurntemp2015'] = hurn_2015_data['Daily Mean Temperature']

# display the head to check it has imported correctly
all_stations_data.head()
# print a summary of the temperature fields for Hurn 1987 and 2015
print(all_stations_data['hurntemp1987'].describe())
print(all_stations_data['hurntemp2015'].describe())
# display a time series plot for the temperatures using the shortdate as the x variable
all_stations_data.plot(x='shortdate', figsize=(12,5))
plt.show()
# import the 1987 data


# import the 2015 data


# add a new column for the 1987 data 


# add a new column for the 2015 data 


# display the head to check it has imported correctly
all_stations_data.head()
# print a summary of the temperature fields for 1987 and 2015

# display a time series plot for the temperatures using the shortdate as the x variable

# print the mean and standard deviation of the temperature for Hurn 1987
print("Hurn 1987 temperature mean: "+str(all_stations_data['hurntemp1987'].mean()))
print("Hurn 1987 temperature standard deviation: "+str(all_stations_data['hurntemp1987'].std()))

# print the mean and standard deviation of the temperature for Hurn 2015
print("Hurn 2015 temperature mean: "+str(all_stations_data['hurntemp2015'].mean()))
print("Hurn 2015 temperature standard deviation: "+str(all_stations_data['hurntemp2015'].std()))
# display a boxplot for the temperatures for 1987 and 2015
all_stations_data.boxplot(column = ['hurntemp1987','hurntemp2015'], vert=False,figsize=(12, 4))
plt.show()
# print the mean and standard deviation of the temperature for 1987


# print the mean and standard deviation of the temperature for 2015


# display a boxplot for the temperatures for 1987 and 2015
