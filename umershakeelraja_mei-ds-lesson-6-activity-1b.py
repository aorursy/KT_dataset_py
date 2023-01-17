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
hurn_all_data.tail()

# check the data types
hurn_all_data.dtypes
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
hurn_jan_data = hurn_all_data[hurn_all_data['mm'] == 1]
# check the head of the data

hurn_jan_data.head()

# plot a time series for the maximum temperature for a single month
hurn_jan_data.plot(x='yyyy', y='tmax', figsize=(12,5))
plt.show()
# create time series for the minimum temperature for some different months

hurn_all_data[(( hurn_all_data['tmin'].str.replace('.', '').str.replace('-','')).str.isnumeric() == False)]
hurn_all_data['tmin'] = hurn_all_data['tmin'].str.replace('*', '')
hurn_all_data['tmin'] = hurn_all_data['tmin'].astype('float')

hurn_jan_data = hurn_all_data[hurn_all_data['mm'] == 1]
hurn_aug_data = hurn_all_data[hurn_all_data['mm'] == 8]

hurn_jan_data.plot(x='yyyy', y='tmin', figsize=(12,5))
hurn_aug_data.plot(x='yyyy', y='tmin', figsize=(12,5))
plt.show()
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


print("Mean of the maximum temperature for January")
print(hurn_jan_data.groupby(['post1989'])['tmax'].mean())

print("\n")


print("Standard deviation of the maximum temperature for January")
print(hurn_jan_data.groupby(['post1989'])['tmax'].std())

hurn_jan_data.boxplot(column = ['tmax'],by='post1989', vert=False,figsize=(10, 5))
plt.title("Maximum temperature for January: Hurn")
plt.show()

# analyse the maximum and minimum temperatures for selected months from one of the other weather stations
heathrow_all_data = pd.read_csv('../input/metofficeweatherbymonth/heathrowdata.csv')
heathrow_all_data = heathrow_all_data.dropna(how='any',axis=0) 


heathrow_all_data['tmax'] = heathrow_all_data['tmax'].astype('str')

heathrow_all_data[(( heathrow_all_data['tmax'].str.replace('.', '').str.replace('-','')).str.isnumeric() == False)]
heathrow_all_data['tmax'] = heathrow_all_data['tmax'].astype('float')

heathrow_all_data['tmin'] = heathrow_all_data['tmin'].astype('str')

heathrow_all_data[(( heathrow_all_data['tmin'].str.replace('.', '').str.replace('-','')).str.isnumeric() == False)]
heathrow_all_data['tmin'] = heathrow_all_data['tmin'].astype('float')


heathrow_jan_data = heathrow_all_data[heathrow_all_data['mm'] == 1]
heathrow_aug_data = heathrow_all_data[heathrow_all_data['mm'] == 8]


heathrow_all_data['post1989'] = heathrow_all_data['yyyy']>1989

print("Mean of the maximum temperature for January")
print(heathrow_jan_data.groupby(['post1989'])['tmax'].mean())

print("\n")


print("Standard deviation of the maximum temperature for January")
print(heathrow_jan_data.groupby(['post1989'])['tmax'].std())


heathrow_jan_data.plot(x='yyyy', y='tmin', figsize=(12,5))
heathrow_aug_data.plot(x='yyyy', y='tmin', figsize=(12,5))

print("Standard deviation of the maximum temperature for August")
print(heathrow_aug_data.groupby(['post1989'])['tmax'].mean())
print("Standard deviation of the maximum temperature for January")
print(heathrow_aug_data.groupby(['post1989'])['tmax'].std())

heathrow_aug_data.boxplot(column = ['tmax'],by='post1989', vert=False,figsize=(10, 5))
plt.title("Maximum temperature for August: Heathrow")
heathrow_jan_data.boxplot(column = ['tmax'],by='post1989', vert=False,figsize=(10, 5))
plt.title("Maximum temperature for January: Heathrow")
plt.show()
