import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_json("../input/train.json")

test = pd.read_json("../input/test.json")
df_all = train.append(test)

df_all["data"] = ["train"]*train.shape[0] + ["test"]*test.shape[0]
df_all.data.value_counts()
train.listing_id.nunique(), test.listing_id.nunique()
np.intersect1d(train.listing_id.unique(), test.listing_id.unique()).size
train.index[:5]
test.index[:5]
ax = df_all.reset_index().groupby("data")["index"].hist(bins=50)
df_all['created'] = pd.to_datetime(df_all['created'])
ax = df_all.groupby(["created", "data"]).price.count().unstack().resample('D').sum().plot(figsize=(9,5))
df_all.groupby("data").price.quantile([0, 0.05, .25, .5, .75, .9, .95, .99, 1]).unstack().T
ax = df_all[df_all.price<13000].groupby("data").price.hist(bins=50)
df_manager = df_all.groupby(["manager_id", "data"]).price.count().unstack().fillna(0).astype(int)

df_manager.sort_values('train', ascending=False).head(10)
df_building = df_all.groupby(["building_id", "data"]).price.count().unstack().fillna(0).astype(int)

df_building.sort_values('train', ascending=False).head(10)
df_price = df_all.groupby(["price", "data"]).price.count().unstack().fillna(0).astype(int)

df_price.sort_values("train", ascending=False).head(10)
df_all.groupby(["bathrooms", "data"]).price.count().unstack().fillna(0).astype(int)
df_all.groupby(["bedrooms", "data"]).price.count().unstack().fillna(0).astype(int)
df_latitude = df_all.groupby(["latitude", "data"]).price.count().unstack().fillna(0).astype(int)

df_latitude.sort_values("train", ascending=False).head(10)
df_longitude = df_all.groupby(["longitude", "data"]).price.count().unstack().fillna(0).astype(int)

df_longitude.sort_values("train", ascending=False).head(10)