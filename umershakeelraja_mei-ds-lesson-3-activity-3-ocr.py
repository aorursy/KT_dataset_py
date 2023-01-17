import pandas as pd
import matplotlib.pyplot as plt

# importing the data
travel_2011_data=pd.read_csv('../input/ocrlds/OCR-lds-travel-2011.csv')

# inspecting the dataset to check that it has imported correctly
travel_2011_data.head()
# check the datatypes

#print(travel_2011_data.dtypes)

# use describe for any fields you are going to investigate and filter out or replace any unusable values

travel_2011_data=travel_2011_data.dropna()
travel_2011_data['Train']=travel_2011_data['Train'].str.replace(',', '').astype('float')
travel_2011_data['Underground, tram']=travel_2011_data['Underground, tram'].str.replace(',', '').astype('float')
travel_2011_data['Bus']=travel_2011_data['Bus'].str.replace(',', '').astype('float')
travel_2011_data['Taxi']=travel_2011_data['Taxi'].str.replace(',', '').astype('float')
travel_2011_data['Motorcycle']=travel_2011_data['Motorcycle'].str.replace(',', '').astype('float')
travel_2011_data['Driving a car']=travel_2011_data['Driving a car'].str.replace(',', '').astype('float')
travel_2011_data['Passenger in a car']=travel_2011_data['Passenger in a car'].str.replace(',', '').astype('float')
travel_2011_data['On foot']=travel_2011_data['On foot'].str.replace(',', '').astype('float')
travel_2011_data['Bicycle']=travel_2011_data['Bicycle'].str.replace(',', '').astype('float')
travel_2011_data['Other']=travel_2011_data['Other'].str.replace(',', '').astype('float')
travel_2011_data['In employment']=travel_2011_data['In employment'].str.replace(',', '').astype('float')
travel_2011_data['Not in employment']=travel_2011_data['Not in employment'].str.replace(',', '').astype('float')


travel_2011_data.head()
#print(travel_2011_data['Train'].describe())

travel_2011_data['Bicycle percent']=travel_2011_data['Bicycle']/travel_2011_data['In employment']*100
travel_2011_data['Underground, tram percent']=travel_2011_data['Underground, tram']/travel_2011_data['In employment']*100
travel_2011_data['Bus percent']=travel_2011_data['Bus']/travel_2011_data['In employment']*100
travel_2011_data['Taxi percent']=travel_2011_data['Taxi']/travel_2011_data['In employment']*100
travel_2011_data['Motorcycle percent']=travel_2011_data['Motorcycle']/travel_2011_data['In employment']*100
travel_2011_data['Driving  a car percent']=travel_2011_data['Driving a car']/travel_2011_data['In employment']*100
travel_2011_data['Passenger in a car percent']=travel_2011_data['Passenger in a car']/travel_2011_data['In employment']*100
travel_2011_data['On foot percent']=travel_2011_data['On foot']/travel_2011_data['In employment']*100
travel_2011_data['Other percent']=travel_2011_data['Other']/travel_2011_data['In employment']*100


# find the means and standard deviations for different fields grouped by region
travel_2011_data['Population']=travel_2011_data['In employment']+travel_2011_data['Not in employment']
travel_2011_data['Bicycle percent']=(travel_2011_data['Bicycle']/travel_2011_data['Population'])*100
travel_2011_data['Underground, tram percent']=(travel_2011_data['Underground, tram']/travel_2011_data['Population'])*100
travel_2011_data['Bus percent']=(travel_2011_data['Bus']/travel_2011_data['Population'])*100
travel_2011_data['Taxi percent']=(travel_2011_data['Taxi']/travel_2011_data['Population'])*100
travel_2011_data['Motorcycle percent']=(travel_2011_data['Motorcycle']/travel_2011_data['Population'])*100
travel_2011_data['Driving a car percent']=(travel_2011_data['Bicycle']/travel_2011_data['Population'])*100
travel_2011_data['Passenger in a car percent']=(travel_2011_data['Passenger in a car']/travel_2011_data['Population'])*100
travel_2011_data['On foot percent']=(travel_2011_data['On foot']/travel_2011_data['Population'])*100
travel_2011_data['Other percent']=(travel_2011_data['Other']/travel_2011_data['Population'])*100

print("Bicycle mean:"+str(travel_2011_data.groupby(['Region'])['Bicycle percent'].mean()))
print("Underground, tram mean:"+str(travel_2011_data.groupby(['Region'])['Underground, tram percent'].mean()))
print("Bus mean:"+str(travel_2011_data.groupby(['Region'])['Bus percent'].mean()))
print("Taxi mean:"+str(travel_2011_data.groupby(['Region'])['Taxi percent'].mean()))
print("Motorcycle mean:"+str(travel_2011_data.groupby(['Region'])['Motorcycle percent'].mean()))
print("Driving a car mean:"+str(travel_2011_data.groupby(['Region'])['Driving a car percent'].mean()))
print("Passenger in a car mean:"+str(travel_2011_data.groupby(['Region'])['Passenger in a car percent'].mean()))
print("On foot mean:"+str(travel_2011_data.groupby(['Region'])['On foot percent'].mean()))
print("Other mean:"+str(travel_2011_data.groupby(['Region'])['Other percent'].mean()))

print("Bicycle std:"+str(travel_2011_data.groupby(['Region'])['Bicycle percent'].std()))
print("Underground, tram std:"+str(travel_2011_data.groupby(['Region'])['Underground, tram percent'].std()))
print("Bus std:"+str(travel_2011_data.groupby(['Region'])['Bus percent'].std()))
print("Taxi std:"+str(travel_2011_data.groupby(['Region'])['Taxi percent'].std()))
print("Motorcycle std:"+str(travel_2011_data.groupby(['Region'])['Motorcycle percent'].std()))
print("Driving a car std:"+str(travel_2011_data.groupby(['Region'])['Driving a car percent'].std()))
print("Passenger in a car std:"+str(travel_2011_data.groupby(['Region'])['Passenger in a car percent'].std()))
print("On foot std:"+str(travel_2011_data.groupby(['Region'])['On foot percent'].std()))
print("Other std:"+str(travel_2011_data.groupby(['Region'])['Other percent'].std()))
#change 'Bus' to other modes of transport
travel_2011_data.boxplot(column = ['Bus'],by='Region', vert=False,figsize=(25, 16))
plt.show()
#change 'Bus' to other modes of transport, bins is number of bars, use bins=[1000, 2000, 3000] etc to set values, use density parameter to normalise
travel_2011_data['Bus'].plot.hist(bins=10, density=0)
plt.show()
#change 'Bus' to other modes of transport
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 30
fig_size[1] = 15
travel_2011_data.plot.scatter(x='Region',y='Bus')

plt.show()
#change 'Bus' or 'Driving a car' to other modes of transport
travel_2011_data.plot.hexbin(x='Driving a car', y='Bus',gridsize=100)
plt.show()
#change 'Bus' to other modes of transport
travel_2011_data.plot(x='Region',y=['Bus'],
                  figsize=(50,10))
plt.show()