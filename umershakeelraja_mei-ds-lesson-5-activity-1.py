import pandas as pd

# import matplotlib
import matplotlib.pyplot as plt

# importing the data
cars_data=pd.read_csv('../input/aqalds/AQA-large-data-set.csv')

# inspecting the dataset to check that it has imported correctly
cars_data.head()
# Explore the shape of data set and get the data types
cars_data.dtypes
cars_data=cars_data[cars_data['Mass'] >0]
cars_data=cars_data[cars_data['EngineSize'] >0]
cars_data['PropulsionTypeId'] = cars_data['PropulsionTypeId'].replace({1: 'Petrol',
                                                                       2: 'Diesel',
                                                                       3: 'Electric', 
                                                                       7: 'Gas/Petrol', 
                                                                       8: 'Electric/Petrol'})
cars_data.head()
# Get a summary of the CO2 field
cars_data['NOX'].describe()
cars_data.boxplot(column = ['CO'],by='PropulsionTypeId', vert=False,figsize=(12, 8))
plt.show()
# Plot the scatter diagram
cars_data.plot.scatter(x='EngineSize', y='CO', figsize=(10,10))
plt.show()
# Plot the scatter diagram
cars_data.plot.scatter(x='Mass', y='NOX', figsize=(10,10))
plt.show()
# plot CO2 against mass with the colour determined by the engine size
cars_data.plot.scatter(x='EngineSize', y='CO2', c='Mass', figsize=(10,10), sharex=False)
plt.show()
# plot CO2 against mass with the size determined by the engine size 
cars_data.plot.scatter(x='EngineSize', y='CO2', s=cars_data['Mass']/200, figsize=(10,10))
plt.show()
# create a mapping for the colours
cmap = {'Petrol': 'red', 'Diesel': 'blue'}

# plot the scatter diagram with the colour set by the mapping
cars_data.plot.scatter(x='Mass', y='CO2', figsize=(10,10),  c=[cmap.get(c, 'black') for c in cars_data.PropulsionTypeId])
plt.show()
cmap = {2002: 'green', 2016:'orange'}
cars_data.plot.scatter(x='Mass', y='CO2', figsize=(10,10),  c=[cmap.get(c, 'black') for c in cars_data.YearRegistered])
plt.show()