import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("hls")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
target = "high_value"
reg_target = "median_house_value"
raw = pd.read_csv("../input/housing_data.csv", index_col=0)
print("N = {} records, {} records with missing values".format(len(raw), len(raw) - len(raw.dropna())))
raw.head()
res = raw.query("population >= 2000 and households >= 1000")
print("Matching Records = {}".format(len(res)))
print("Proportion High Value = {0:.3f}".format(res[target].mean()))
res.head()
train = raw.query("split == 'train'")
ax = sns.scatterplot(x="longitude", y="latitude", hue="median_house_value", data=train)
ax.figure.set_size_inches(6, 6)
plt.title("Heatmap of District Median House Values")
plt.show()
ocean_values = raw["ocean_proximity"].value_counts()
sns.barplot(ocean_values.index, ocean_values.values)
plt.xlabel("Ocean Proximity")
plt.ylabel("Count")
plt.title("Values for Ocean Proximity")
plt.show()
sns.distplot(raw["median_income"], kde=False, bins=20)
plt.ylabel("Count")
plt.title("Distribution of District Median Household Income")
plt.show()
