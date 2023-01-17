# import pandas
import pandas as pd 

# import matplotlib
import matplotlib.pyplot as plt
hurn_all_data = pd.read_csv('../input/metofficeweatherbymonth/hurndata.csv')
hurn_all_data.head()
# create a new column that will be True or False depending on the statement: heathrow_aug_data['yyyy']>1989
hurn_all_data['post1989'] = hurn_all_data['yyyy']>1989

# check the data by displaying the first few rows
hurn_all_data.head()
# check the last rows of the dataset

# check the data types

# search for any rows where the temperature is not a viable number
hurn_all_data[(( hurn_all_data['tmax'].str.replace('.', '')).str.isnumeric() == False)]
# remove '*' signs by replacing them with an empty string
hurn_all_data['tmax'] = hurn_all_data['tmax'].str.replace('*', '')

# convert to float type
hurn_all_data['tmax'] = hurn_all_data['tmax'].astype('float')

# check the data types
hurn_all_data['tmax'].describe()
# plot a time series for the data
hurn_all_data.plot(y='tmax', figsize=(10,5))
plt.show()
# create a new data set with just the august data
hurn_aug_data = hurn_all_data[hurn_all_data['mm'] == 8]

# check the head of the data
hurn_aug_data.head()
# plot a time series for the maximum temperature for august
hurn_aug_data.plot(x='yyyy', y='tmax', figsize=(12,5))
plt.show()
# create a new data set with the data for a single month


# check the head of the data


# plot a time series for the maximum temperature for a single month

# create time series for the minimum temperature for some different months
# print the means
print("Mean of the maximum temperature for August")
print(hurn_aug_data.groupby(['post1989'])['tmax'].mean())

# print a blank line
print("\n")

# print the standard deviation
print("Standard deviation of the maximum temperature for August")
print(hurn_aug_data.groupby(['post1989'])['tmax'].std())

# display the boxplot
hurn_aug_data.boxplot(column = ['tmax'],by='post1989', vert=False,figsize=(10, 5))
plt.title("Maximum temperature for August: Hurn")
plt.show()
# display means, standard deviations and boxplots for maximum and minimum temperature for the other months
# analyse the maximum and minimum temperatures for selected months from one of the other weather stations