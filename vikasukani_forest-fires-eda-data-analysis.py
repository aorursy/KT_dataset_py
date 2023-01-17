# Install Necessary Packages, (First Time Only)



# !pip install numpy --upgrade

# !pip install pandas --upgrade

# !pip install matplotlib --upgrade

# !pip install seaborn --upgrade



import os



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





%matplotlib inline

sns.set_style('darkgrid')
os.listdir('../input/forest-firearea-datasets')
# Reading csv file



forest_df = pd.read_csv('../input/forest-firearea-datasets/forestfires.csv', delimiter=',',  )

#  Display the first few rows



forest_df.head(20)
forest_df.shape
feature_names = forest_df.columns



feature_names
forest_df.info()
# Display the object type of data information.



forest_df.describe(include=['O'])
# Lets create an variable and store its column name



categorical_feature = forest_df.describe(include=['O']).columns



# Print it in list type ...

print(list(categorical_feature))
forest_df['month'].value_counts()
forest_df['day'].value_counts()
forest_df.describe(include=["int", "float"])
numerical_feature = forest_df.describe(include=["int", "float"]).columns



# Print it in list type ..

print(list(numerical_feature))
forest_df['area_km'] = forest_df['area'] / 100



forest_df
# Shows the highes area in km



forest_df.sort_values(by="area_km", ascending=False).head()
highest_fire_area = forest_df.sort_values(by="area_km", ascending=True)



plt.figure(figsize=(8, 6))



plt.title("Temperature vs area of fire" )

plt.bar(highest_fire_area['temp'], highest_fire_area['area_km'])



plt.xlabel("Temperature")

plt.ylabel("Area per km-sq")

plt.show()
print(categorical_feature)





plt.figure(figsize=(10, 5))

for idx, column in enumerate(categorical_feature):

    df = forest_df.copy()

    unique = df[column].value_counts(ascending=True);

 

    plt.subplot(1, 2, idx+1)    

    plt.title("Count of "+ column)

    plt.bar(unique.index, unique.values);

    

    plt.xlabel(column)

    plt.ylabel("Number of "+ column)

    

plt.tight_layout()

plt.show()     
len(numerical_feature)



# forest_df[numerical_feature]

forest_df[categorical_feature]
sns.set_style('darkgrid')

# Find the relation

# plt.subplot(forest_df)



sns.pairplot(forest_df[["temp", "wind", "rain", "area_km"]])

plt.show()

for idx, col  in enumerate(numerical_feature, 1):

    plt.figure(figsize=(5,5))

    

#     plt.subplot(len(numerical_feature) // 2, 3, idx)

    plt.boxplot(forest_df[col])



    plt.title(col)

#     plt.hist(forest_df[col])



plt.tight_layout()

plt.show(plt)
plt.figure(figsize=(15, 12))



plt.title("Heatmap Relation")



sns.heatmap(forest_df[numerical_feature].corr(), annot=True, fmt='.2f');

forest_df
plt.figure(figsize=(10, 7))



plt.scatter(forest_df['X'], forest_df['area_km'])

plt.scatter(forest_df['Y'], forest_df['area_km'])





plt.show()


highest_rain = forest_df.sort_values(by='rain', ascending=False)[['month', 'day', 'rain']].head()

highest_rain

highest_temp = forest_df.sort_values(by='temp', ascending=False)[['month', 'day', 'temp']].head()



lowest_temp =  forest_df.sort_values(by='temp', ascending=True)[['month', 'day', 'temp']].head()



print("Highest Temperature")



print(highest_temp)



print()



print()



print("Lowest Temperature")

print(lowest_temp)

plt.figure(figsize=(9, 6))



plt.title("Highest Temperature in Aug.  Month")



plt.bar(highest_temp['day'], highest_temp['temp'])



plt.xlabel("Day")

plt.xlabel("Aug. Month")





plt.ylabel("Temperature")

plt.show()
plt.figure(figsize=(9, 6))



plt.title("Lowest Temperature in Dec. and Feb.  Month")



plt.bar(lowest_temp['day'], lowest_temp['temp'])



plt.xlabel("Day")

plt.xlabel("Dec. and Feb. Month")



plt.ylabel("Temperature")



plt.show()
forest_df