# import libraries

import pandas as pd
vehicle = pd.read_csv('/kaggle/input/automobile-dataset/Automobile_data.csv')

vehicle.head()
vehicle.tail()
vehicle.head(25)
vehicle.info()
vehicle.describe()
print("vehicle:-  ", vehicle.shape)

print("Rows:-     ", vehicle.shape[0])

print("Columns:-  ", vehicle.shape[1])

print("Features:- \n", vehicle.columns.tolist())

print("\nMissing Values:- ", vehicle.isnull().values.sum())

print("\nUnique Values:- \n", vehicle.nunique())
print(vehicle.dtypes)
print("Unique values for make:- " , vehicle.make.unique())

print("\nUnique values for num-of-doors:- " , vehicle['num-of-doors'].unique())  

print("\nUnique values for body-style :- " , vehicle['body-style'].unique())

print("\nUnique values for drive-wheels:- " , vehicle['drive-wheels'].unique())

print("\nUnique values for engine-location:-", vehicle['engine-location'].unique())

print("\nUnique values for bore:-", vehicle.bore.unique())

print(vehicle.dtypes)
df = vehicle[vehicle["horsepower"] == '?']

df
#Replacing missing values of horsepower with mean of horsepower

horsepower = vehicle['horsepower'].loc[vehicle["horsepower"] != '?']

hpmean = horsepower.astype(str).astype(int).mean()

vehicle['horsepower'] = vehicle['horsepower'].replace("?", hpmean).astype(int)

vehicle['horsepower']
vehicle['peak-rpm'].loc[vehicle['peak-rpm'] == '?']
#Replacing missing values of peak-rpm with mean of peak-rpm 

pick_rpm = vehicle['peak-rpm'].loc[vehicle['peak-rpm'] != '?']

rpm_mean = pick_rpm.astype(str).astype(int).mean()

print("Mean Value of pick-rpm:- "+ str(rpm_mean))

vehicle['peak-rpm'] = vehicle['peak-rpm'].replace('?', rpm_mean).astype(int)

vehicle['peak-rpm']
vehicle['bore'].loc[vehicle['bore'] == '?']
#Replacing missing values of bore with mean of bore 

bore = vehicle['bore'].loc[vehicle['bore'] != '?']

bore_mean = pd.to_numeric(bore, errors='coerce').mean()

print("Mean Value of bore:- "+ str(bore_mean))

vehicle['bore'] = vehicle['bore'].replace('?', bore_mean).astype(float)

vehicle['bore']
vehicle['stroke'].loc[vehicle['stroke'] == '?']
#Replacing missing values of bore with mean of bore 

stroke = vehicle['stroke'].loc[vehicle['stroke'] != '?']

stroke

stroke_mean = pd.to_numeric(stroke, errors='coerce').mean()

print("Mean Value of stroke:- "+ str(stroke_mean))

vehicle['stroke'] = vehicle['stroke'].replace('?', stroke_mean).astype(float)

vehicle['stroke']
vehicle['price'].loc[vehicle['price'] == '?']
#Replacing missing values of price with mean of price 

price = vehicle['price'].loc[vehicle['price'] != '?']

price_mean = price.astype(str).astype(int).mean()

print("Mean Price of all vehicle price:- " +str(price_mean))

vehicle['price'] = vehicle['price'].replace("?", price_mean).astype(int)

vehicle['price']
print(vehicle.dtypes)
vehicle.head(100)