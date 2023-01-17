import pandas as pd

import numpy as np

import scipy as sp

import seaborn as sns

import matplotlib.pyplot as plt# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import scipy as sp

import seaborn as sns

import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
raw_df = pd.read_csv('/kaggle/input/whitewinepricerating/white-wine-price-rating.csv', sep=",")

raw_df.dropna(inplace=True)

raw_df.shape
# remove entries that do not met minimum rating count and rating

MINIMUM_RATING = 4.1

MINIMUM_RATING_COUNT = 30

df = (

    raw_df[

        lambda x: (x.WineRating > MINIMUM_RATING)

        & (x.WineRatingCount > MINIMUM_RATING_COUNT)

    ]

    .groupby("Region")

    .filter(lambda group: group.FullName.size > 30)

    .groupby("Winery")

    .filter(

        lambda group: group.FullName.size > 5

    )  # requires a minium of 5 WINE*VINTAGE

)

df["AdjustedWineRating"] = df["WineRating"] - MINIMUM_RATING

df.shape
# %%

# plot out distributino of VintagePrice to check density

#

# result: heavily concentrated on the head while the tail streches expoentially

#

# indication: as axises of plotting are not tuned logarithmic, cutting into

# bins help to decompress the plot

#

print(df.VintagePrice.describe())

sns.distplot(df[lambda x: x.VintagePrice < 1000].VintagePrice, bins=20)
# %%

# cut data into 3 bins by VintagePrice

# the goal is trying to find data cluster in separate KDE distribution

#

price_bins = pd.cut(

    df["VintagePrice"],

    [0, 50, 100, 1000, 10000],

    labels=["low", "mid", "high", "ultra-high"],

)

df["VintagePriceBin"] = price_bins

price_low_df = df.groupby("VintagePriceBin").get_group("low")

price_mid_df = df.groupby("VintagePriceBin").get_group("mid")

price_high_df = df.groupby("VintagePriceBin").get_group("high")

price_ultra_high_df = df.groupby("VintagePriceBin").get_group("ultra-high")

print(

    [

        price_low_df.shape,

        price_mid_df.shape,

        price_high_df.shape,

        price_ultra_high_df.shape,

    ]

)
# %%

# plotting - check if there's obvious pattern between price/rating

# KDE plot would be easier to read than scatter

# separate KDE plots help to decompress the logarithmic y-scale

#





def kdeplot(input, color):

    return sns.jointplot(

        input.VintageRating,

        input.VintagePrice,

        kind="kde",

        dropna=True,

        xlim=(3.9, 5.0),

        cmap=sns.cubehelix_palette(start=color, light=1, as_cmap=True),

        stat_func=sp.stats.pearsonr,

    )





kdeplot(price_ultra_high_df, 4)

kdeplot(price_high_df, 3)

kdeplot(price_mid_df, 2)

kdeplot(price_low_df, 1)
# %%

# plotting with trendline to find suitable linear model

#

# R-Squared is returned as [1][0] in the polyfit

# the result show that R-Squared is smaller when degree>=3

# given the outliers in the ultra-high price range

#

sns.lmplot(data=df, x="VintageRating", y="VintagePrice", order=1)

sns.lmplot(data=df, x="VintageRating", y="VintagePrice", order=2)

sns.lmplot(data=df, x="VintageRating", y="VintagePrice", order=3)

sns.lmplot(data=df, x="VintageRating", y="VintagePrice", order=4)

print(np.polyfit(df.VintageRating, df.VintagePrice, deg=1, full=True)[1][0])

print(np.polyfit(df.VintageRating, df.VintagePrice, deg=2, full=True)[1][0])

print(np.polyfit(df.VintageRating, df.VintagePrice, deg=3, full=True)[1][0])

print(np.polyfit(df.VintageRating, df.VintagePrice, deg=4, full=True)[1][0])
# %%

# fit different price bins separately with linear models

#

sns.lmplot(data=price_low_df, x="VintageRating", y="VintagePrice", order=1)

sns.lmplot(data=price_mid_df, x="VintageRating", y="VintagePrice", order=1)

sns.lmplot(data=price_high_df, x="VintageRating", y="VintagePrice", order=1)

sns.lmplot(data=price_ultra_high_df, x="VintageRating", y="VintagePrice", order=1)

print(

    np.polyfit(price_low_df.VintageRating, price_low_df.VintagePrice, deg=1, full=True)

)

print(

    np.polyfit(price_mid_df.VintageRating, price_mid_df.VintagePrice, deg=1, full=True)

)

print(

    np.polyfit(

        price_high_df.VintageRating, price_high_df.VintagePrice, deg=1, full=True

    )

)

print(

    np.polyfit(

        price_ultra_high_df.VintageRating,

        price_ultra_high_df.VintagePrice,

        deg=1,

        full=True,

    )

)
# %%

# get the intersect of WinePrice/AdjustedWineRating for each region

# i.e. how much you need to pay to get the MINIMUM_RATING

#

# usage: if you want to buy wines on the MINIMUM_RATING end

# regions with lower intersect have better money-value-performance

#

region_by_price_rating_intersect = (

    df[lambda x: (x.WinePrice < 1000)]

    .groupby("Region")

    .apply(

        lambda region: np.polyfit(region.AdjustedWineRating, region.WinePrice, deg=1)[1]

    )

    .sort_values()

)

region_by_price_rating_intersect
# %%

# get the slope of how fast the price increases with rating in each region

# i.e. how much premium/popular good wines of the region is

# a.k.a how much averagely more you need to pay to get 0.1 better rating

#

# usage: if you want to buy premium wines, which region needs less premium

#

region_by_price_rating_slope = (

    df[lambda x: (x.WinePrice < 1000)]

    .groupby("Region")

    .apply(

        lambda region: np.polyfit(region.AdjustedWineRating, region.WinePrice, deg=1)[0]

    )

    .sort_values()

)

region_by_price_rating_slope
# %%

#

# visualise the slop of price/rating by regions

#

sns.lmplot(

    data=df[lambda x: (x.WinePrice < 1000)],

    x="AdjustedWineRating",

    y="WinePrice",

    hue="Region",

    col="Region",

    col_wrap=3,

    col_order=region_by_price_rating_slope.index,

    height=5,

    order=1,

    ci=None,

)
# %%

# sort regions by its median rating/price ratio

# because price is not normally distributed, prefer median over mean

#

region_by_premium_order = (

    df.groupby("Region").median()["WineRatingPriceRatio"].sort_values(ascending=False)

)



plt.figure(figsize=(10, 4))

ax = sns.barplot(

    x="Region",

    y="WineRatingPriceRatio",

    data=df,

    order=region_by_premium_order.index,

    ci="sd",

)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
# %%

# get the mean and std of WineRating by Region

#

region_rating_order = (

    df.groupby("Region").mean()["WineRating"].sort_values(ascending=False)

)



plt.figure(figsize=(10, 4))

ax = sns.barplot(

    x="Region",

    y="WineRating",

    data=df.transform(lambda x: x - 4 if x.name == "WineRating" else x),

    order=region_rating_order.index,

    ci="sd",

)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
df["AdjustedWineRating"] = df["WineRating"] - MINIMUM_RATING
# %%

# plot the top 30 wineries by WineRating

#

productive_winery = df.groupby("Winery").filter(lambda winery: winery.FullName.size > 5)

productive_winery_by_rating = (

    productive_winery.groupby("Winery")

    .mean()["WineRating"]

    .sort_values(ascending=False)

    .head(30)

)



plt.figure(figsize=(15, 4))

ax = sns.barplot(

    x="Winery",

    y="AdjustedWineRating",

    data=productive_winery,

    order=productive_winery_by_rating.index,

    ci="sd",

)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
# %%

# plot wineries with best WineRatingPriceRatio in popular regions

#





def plotTopValuWinery(region, **premium):

    premium_winery = productive_winery[

        lambda entry: (entry.WineRating > 4.3) & (entry.Region == region)

        if premium

        else (entry.Region == region)

    ]



    winery_order_by_value = (

        premium_winery.groupby("Winery")

        .median()["WineRatingPriceRatio"]

        .sort_values(ascending=False)

        .head(20)

    )



    plt.figure(figsize=(15, 4))

    ax = sns.barplot(

        x="Winery",

        y="WineRatingPriceRatio",

        data=premium_winery,

        order=winery_order_by_value.index,

        ci="sd",

        estimator=np.median,

    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")





plotTopValuWinery("Burgundy", premium=True)

plotTopValuWinery("Burgundy")

plotTopValuWinery("Bordeaux")

plotTopValuWinery("Alsace")

plotTopValuWinery("German")
# %%

# find the top rated Year by Region

#

df.groupby(["Region", "Year"]).filter(lambda group: group.FullName.size > 5).groupby(

    ["Region", "Year"]

).mean()["VintageRating"].sort_values(ascending=False).head(50)