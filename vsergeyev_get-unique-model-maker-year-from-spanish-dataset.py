import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



import os

print(os.listdir("../input/"))



path=r'../input/data.csv'

all_data = pd.read_csv(path, sep=';', encoding='Latin1', low_memory=False)



# Remove non complete data



work_data = all_data[all_data.model != '']

work_data = work_data[work_data.make != '']



#inspecting: make, model

print(len(work_data.groupby('make').size()), 'different makers')

print(len(work_data.groupby('model').size()), 'different models')



work_data = work_data.drop("ID", 1)

work_data = work_data.drop("version", 1)

work_data = work_data.drop("power", 1)

work_data = work_data.drop("sale_type", 1)

work_data = work_data.drop("num_owners", 1)

work_data = work_data.drop("gear_type", 1)

work_data = work_data.drop("fuel_type", 1)

work_data = work_data.drop("kms", 1)

work_data = work_data.drop("price", 1)



work_data = work_data.drop_duplicates()



print(len(work_data), 'rows')



work_data.head(10)
import datetime



data_collected_at = "18/03/2018"

format_str = "%d/%m/%Y"

# dataset has only "months_old" column

# to get Car Manufactured Year we need to calculate it

# e.g. data_collected - months_old

def calc_year_manufactured(months_old):

    try:

        return (datetime.datetime.strptime(data_collected_at, format_str) - datetime.timedelta(months_old*365/12)).year

    except ValueError:

        return 0



work_data["year"] = work_data["months_old"].map(calc_year_manufactured)



work_data = work_data[work_data.year != 0]

print(len(work_data), 'rows with non empty Year of Manufacture')



cleaned_data = work_data.drop("months_old", 1)

cleaned_data = cleaned_data.drop_duplicates()

print(len(cleaned_data), 'rows without duplicates')



#inspecting: year

plt.subplot(3,1,1)

cleaned_data['year'].hist(bins=50)

plt.title('Year of Manufacture')

plt.show()



cleaned_data.head(10)



# save cleaned data

cleaned_data.to_csv("out.csv", sep=',', encoding='utf-8')