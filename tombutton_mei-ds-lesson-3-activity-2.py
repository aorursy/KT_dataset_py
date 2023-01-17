# import pandas
import pandas as pd

# importing the data
cars_data=pd.read_csv('../input/aqalds/AQA-large-data-set.csv')

# inspecting the dataset to check that it has imported correctly
cars_data.head()
print(cars_data['Mass'].describe())
# A new copy of the cars_data dataset created 
# This contains only those values from the original cars_data dataset where the value of the Mass is >0
cars_data=cars_data[cars_data['Mass'] >0]

# describe is used to check that these values look suitable
print(cars_data['Mass'].describe())
# the PropulsionTypeId field is overwritten with the values from the list using the replace command
# the replace list uses a colon to indicate what is to be replaced and a commas to separate the items
cars_data['PropulsionTypeId'] = cars_data['PropulsionTypeId'].replace({1: 'Petrol',
                                                                       2: 'Diesel',
                                                                       3: 'Electric', 
                                                                       7: 'Gas/Petrol', 
                                                                       8: 'Electric/Petrol'})
cars_data.head()
# import matplotlib
import matplotlib.pyplot as plt
cars_data.boxplot(column = ['Mass'],by='PropulsionTypeId', vert=False,figsize=(12, 8))
plt.show()

cars_data.boxplot(column = ['CO2'],by='PropulsionTypeId', vert=False,figsize=(12, 8))
plt.show()
# NOTE: This block of code will only work if the PropulsionTypeId has been changed to the text values

# a new datset called petrol_data is created that contains only the rows where the Propulsion Type is Petrol
petrol_data = cars_data[cars_data['PropulsionTypeId'] == 'Petrol']
petrol_data.head()
# print out the mean and standard deviation for the Mass converted to text (i.e. a string)
print("mass: mean = "+str(petrol_data['Mass'].mean()))
print("mass: standard deviation = "+str(petrol_data['Mass'].std()))

# NOTE: This block of code will only work if the PropulsionTypeId has been changed to the text values

# a new datset called petrol_data is created
# this contains only the rows where the Propulsion Type is Petrol
petrol_data = cars_data[cars_data['PropulsionTypeId'] == 'Petrol']

# print out the mean and standard deviation for the Mass converted to text (i.e. a string)
print("mass: mean = "+str(petrol_data['Mass'].mean()))
print("mass: standard deviation = "+str(petrol_data['Mass'].std()))