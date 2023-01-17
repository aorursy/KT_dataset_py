import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("setup complete")



sns.set_style('whitegrid')

ev_data = pd.read_csv("../input/public-charging-stations-for-cars-in-hawaii/Public_Charging_Stations_in_Hawaii.csv")



ev_data.head()
ev_data.info()
ev_data.describe()
ev_data.Island.unique()
def Series(Island):

    series = ''

    for char in Island:

        if char.isalpha():

            series += char

    return series



def Sisland(Island):

    sisland = ''

    for char in Island:

        if char.isdigit():

            sisland += char

    if sisland == '':

        return 1

    return int(sisland)
ev_data['series']=ev_data['Island'].apply(Series)

ev_data['sisland']=ev_data['Island'].apply(Sisland)