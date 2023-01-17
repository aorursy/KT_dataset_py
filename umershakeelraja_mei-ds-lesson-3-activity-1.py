# import pandas module
import pandas as pd

# importing the data
travel_2011_data=pd.read_csv('../input/ocrlds/OCR-lds-travel-2011.csv')

# inspecting the dataset to check that it has imported correctly
travel_2011_data.head()
# explore the data
print(travel_2011_data.dtypes)

print(travel_2011_data.shape)
# any commas in the In employment fields are removed by replacing them with an empty string
travel_2011_data['In employment'] = travel_2011_data['In employment'].str.replace(',', '')

# the fields are then convert to the float type
travel_2011_data['In employment'] = travel_2011_data['In employment'].astype('float')

# you can then use the describe command to check that this has worked and they can be analysed
travel_2011_data['In employment'].describe()
# any commas in the In employment fields are removed by replacing them with an empty string
travel_2011_data['On foot'] = travel_2011_data['On foot'].str.replace(',', '')

# the fields are then convert to the float type
travel_2011_data['On foot'] = travel_2011_data['On foot'].astype('float')

# you can then use the describe command to check that this has worked and they can be analysed
travel_2011_data['On foot'].describe()
# The percentage is calculated and stored in a new field: Bicycle percent
travel_2011_data['Bicycle percent']=travel_2011_data['Bicycle']/travel_2011_data['In employment']*100
travel_2011_data['On foot percent']=travel_2011_data['On foot']/travel_2011_data['In employment']*100
travel_2011_data.head()
# import matplotlib for plotting
import matplotlib.pyplot as plt
# plot a boxplot for Bicycle percent grouped by Region
travel_2011_data.boxplot(column = ['Bicycle percent'],by='Region', vert=False, figsize=(12, 8))
plt.show()
print(travel_2011_data.groupby(['Region'])['Bicycle percent'].mean())

print(travel_2011_data.groupby(['Region'])['Bicycle percent'].std())

print(travel_2011_data.groupby(['Region'])['On foot percent'].mean())

print(travel_2011_data.groupby(['Region'])['On foot percent'].std())