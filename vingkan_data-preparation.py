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
data = pd.DataFrame(raw.query("split == 'train'")).drop(columns=["split"])
data.head()
ocean_prox_counts = data["ocean_proximity"].value_counts()
sns.barplot(ocean_prox_counts.index, ocean_prox_counts.values)
plt.show()
data["ocean_proximity"] = ["within_hour_ocean" if val == "<1H OCEAN" else val for val in data["ocean_proximity"]]
data["ocean_proximity"] = data["ocean_proximity"].apply(lambda x: x.lower().replace(" ", "_"))
ax = sns.scatterplot(x="longitude", y="latitude", hue="ocean_proximity", data=data)
ax.figure.set_size_inches(6, 6)
data = pd.get_dummies(data, columns=["ocean_proximity"])
data.head().T
for p in data.columns:
    n_missing = len(data) - len(data.dropna(subset=[p]))
    print("{} records with missing value for {}".format(n_missing, p))
sns.distplot(data["total_bedrooms"].dropna(), kde=False, bins=20)
null_data = data[data.isna().any(axis=1)]
ax = sns.scatterplot(x="longitude", y="latitude", color="lightgray", data=data)
ax = sns.scatterplot(x="longitude", y="latitude", hue="median_house_value", palette="Greens", data=null_data)
plt.title("Median House Values of Missing Districts")
ax.figure.set_size_inches(6, 6)
from sklearn.impute import SimpleImputer

mean_imp = SimpleImputer(strategy="mean")
median_imp = SimpleImputer(strategy="median")
mode_imp = SimpleImputer(strategy="most_frequent")

data["total_bedrooms_mean_imp"] = mean_imp.fit_transform(data["total_bedrooms"].values.reshape(-1, 1))
data["total_bedrooms_median_imp"] = median_imp.fit_transform(data["total_bedrooms"].values.reshape(-1, 1))
data["total_bedrooms_mode_imp"] = mode_imp.fit_transform(data["total_bedrooms"].values.reshape(-1, 1))

bedroom_features = [
    "total_bedrooms",
    "total_bedrooms_mean_imp",
    "total_bedrooms_median_imp",
    "total_bedrooms_mode_imp"
]
null_imputed = data[data.isna().any(axis=1)]
null_imputed[bedroom_features].head()
sf_lat = 37.756460
sf_lon = -122.442749
sf_dist = np.sqrt((data["latitude"] - sf_lat) ** 2 + (data["longitude"] - sf_lon) ** 2)
data["proximity_sf"] = sf_dist.max() - sf_dist
ax = sns.scatterplot(x="longitude", y="latitude", hue="proximity_sf", palette="Reds", data=data)
plt.title("Heatmap of Districts by Distance to San Francisco")
ax.figure.set_size_inches(6, 6)
la_lat = 34.121552
la_lon = -118.360661
la_dist = np.sqrt((data["latitude"] - la_lat) ** 2 + (data["longitude"] - la_lon) ** 2)
data["proximity_la"] = la_dist.max() - la_dist
ax = sns.scatterplot(x="longitude", y="latitude", hue="proximity_la", palette="Blues", data=data)
plt.title("Heatmap of Districts by Distance to Los Angeles")
ax.figure.set_size_inches(6, 6)
def plot_scaler_row(cleaned, feature, scalers, pal):
    """Helper function to visualize the effect of different scalers."""
    fig, row = plt.subplots(1, len(scalers), sharey=True)
    for color, col, scaler in zip(pal, row, scalers):
        arr = cleaned[feature].values.reshape(-1, 1)
        fitted = scaler.fit_transform(arr)
        col.hist(fitted, color=color)
        col.set_xlabel(type(scaler).__name__)
        col.set_ylabel(feature)
    fig.set_size_inches(2 * len(scalers), 2)
    for ax in row.flat:
        ax.label_outer()
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

cont_features = [
    "housing_median_age",
    "median_income",
    "population",
    "total_bedrooms_mean_imp",
    "total_bedrooms_median_imp",
    "total_bedrooms_mode_imp",
    "total_rooms",
    "households",
    "proximity_sf",
    "proximity_la"
]
scalers = [
    StandardScaler(),
    MinMaxScaler(),
    RobustScaler(),
    PowerTransformer(),
    QuantileTransformer(output_distribution="normal"),
    QuantileTransformer(output_distribution="uniform")
]
pal = sns.color_palette("Blues_d", len(scalers))
for feature in cont_features:
    plot_scaler_row(data, feature, scalers, pal)
for feature in cont_features:
    scaler = QuantileTransformer(output_distribution="normal")
    data[feature + "_scaled"] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
data.head().T
data.describe().T
