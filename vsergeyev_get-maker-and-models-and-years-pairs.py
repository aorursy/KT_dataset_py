# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



import os

print(os.listdir("../input/"))



path=r'../input/all_anonymized_2015_11_2017_03.csv'

all_data = pd.read_csv(path, sep=',',encoding='Latin1', low_memory=False)



work_data = all_data[all_data.model != '']



work_data = work_data[work_data.maker != '']

work_data = work_data[work_data.maker.isnull() == False]

work_data = work_data[work_data.model.isnull() == False]



#inspecting: maker

print(len(work_data.groupby('maker').size()), 'different makers')

print(len(work_data.groupby('model').size()), 'different model')



work_data = work_data.drop("mileage", 1)

# work_data = work_data.drop("engine_displacement", 1)

work_data = work_data.drop("engine_power", 1)

work_data = work_data.drop("body_type", 1)

work_data = work_data.drop("color_slug", 1)

work_data = work_data.drop("stk_year", 1)

# work_data = work_data.drop("transmission", 1)

work_data = work_data.drop("door_count", 1)

work_data = work_data.drop("seat_count", 1)

# work_data = work_data.drop("fuel_type", 1)

work_data = work_data.drop("date_created", 1)

work_data = work_data.drop("date_last_seen", 1)

work_data = work_data.drop("price_eur", 1)



work_data = work_data.drop_duplicates()



work_data.head(10)
print(len(work_data), 'rows')



work_data["manufacture_year"] = work_data["manufacture_year"].apply(pd.to_numeric, errors='coerce', downcast='integer')



print(len(work_data.groupby(['maker', 'model', 'manufacture_year', 'engine_displacement', 'transmission', 'fuel_type']).size()), 'unique combinations')



cleaned_data = work_data



cleaned_data.head(10)



cleaned_data.to_csv("out.csv", sep=',', encoding='utf-8')