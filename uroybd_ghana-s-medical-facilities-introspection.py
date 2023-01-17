# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg 

import seaborn as sns

plt.figure(figsize=(15,8.5))

sns.set(style="whitegrid", rc={'figure.figsize':(15,8.5)})

sns.set_palette("Spectral")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



facilities_file = "../input/health-facilities-gh/health-facilities-gh.csv"

tier_file = "../input/health-facilities-gh/health-facility-tiers.csv"



facilities_data = pd.read_csv(facilities_file)

tier_data = pd.read_csv(tier_file)

# Any results you write to the current directory are saved as output.
print(facilities_data.columns)

facilities_data.head()
print(tier_data.columns)

tier_data.head()
facilities_data['FacilityName'] = facilities_data['FacilityName'].apply(lambda x: x.upper())

facilities_data['Ownership'] = facilities_data['Ownership'].apply(lambda x: x.capitalize())
combined_data = pd.merge(facilities_data,  tier_data, how="left", left_on=["Region", "FacilityName"], right_on=["Region", "Facility"])

del combined_data["Facility"]

print(combined_data.columns)

combined_data.head()
# combined_data.plot.pie(y="Ownership")

ownerships = combined_data["Ownership"].value_counts(normalize=True)*100

indexs = ownerships.index

ownerships.drop(indexs[2:], inplace=True)

ownerships["Others"] = 100 - ownerships[:3].sum()

ownerships.plot.pie(figsize=(10, 10), autopct='%1.1f%%')
regions = combined_data["Region"].value_counts(normalize=True)*100

regions.plot.pie(figsize=(10, 10), autopct='%1.1f%%')
tiers = tier_data.fillna(value={"Tier": 0})["Tier"].value_counts()

tiers.plot.pie(figsize=(10, 10), autopct='%1.1f%%')
region_govt_ownership_data_series = combined_data.loc[combined_data["Ownership"] == "Government"].groupby("Region")["Ownership"].count()

sns.barplot(region_govt_ownership_data_series.index, region_govt_ownership_data_series.values)
region_private_ownership_data_series = combined_data.loc[combined_data["Ownership"] == "Private"].groupby("Region")["Ownership"].count()

sns.barplot(region_private_ownership_data_series.index, region_private_ownership_data_series.values)
other_institutions = combined_data[~combined_data.Ownership.isin(["Private", "Government"])]

other_institutions_counts = other_institutions['Ownership'].value_counts()

other_institutions_counts

sns.barplot(other_institutions_counts.index, other_institutions_counts.values)
region_other_ownership_data_series = other_institutions.groupby("Region")["Ownership"].count()

sns.barplot(region_other_ownership_data_series.index, region_other_ownership_data_series.values)
ghana_map_file = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Ghana_location_map.svg/500px-Ghana_location_map.svg.png"

map_img = mpimg.imread(ghana_map_file) 

cax = sns.scatterplot(x="Longitude", y="Latitude", data=combined_data.fillna(value={"Tier": 0}), hue="Tier", palette="Spectral_r")

cax.axis("equal")

cax.imshow(map_img, zorder=0, extent=[-3.6, 1.5, 4.5, 11.4])
cax = sns.scatterplot(x="Longitude", y="Latitude", data=combined_data, hue="Ownership", palette="Set1")

cax.axis("equal")

cax.imshow(map_img, zorder=0, extent=[-3.6, 1.5, 4.5, 11.4])
total = len(combined_data.Town)

urban = combined_data.Town.count()

rural = total - urban

fig1, ax1 = plt.subplots()

ax1.pie([urban, rural], labels=["Urban", "Rural"], autopct='%1.1f%%')

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()