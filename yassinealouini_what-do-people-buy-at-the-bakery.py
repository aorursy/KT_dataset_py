# The usual imports
import pandas as pd
import missingno as msno
import pandas_profiling as pdp
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pylab as plt
%matplotlib inline
DATA_PATH = "../input/BreadBasket_DMS.csv"
df = pd.read_csv(DATA_PATH)
df.sample(5)
print(f"The data start at {df.Date.min()} and ends at {df.Date.max()}")
pdp.ProfileReport(df)
msno.matrix(df)
# Construct a timestamp colum using the Data and Time ones, then drop them.
df["tms"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.drop(["Date", "Time"], axis=1)
# To answer this question, I will create a word cloud.
w = WordCloud().generate_from_frequencies(df.Item.value_counts().to_dict())
plt.imshow(w)
df.Item.value_counts().nlargest(5).plot(kind='bar')
df.groupby(df.tms.dt.year).Item.value_counts().nlargest(2).plot(kind='bar')
(df.groupby(df.Transaction)
   .Item.count()
   .value_counts(normalize=True)
   .mul(100).plot(kind='bar'))
df.groupby(df.tms.dt.date).Transaction.nunique().plot()
(df.assign(dow=df.tms.dt.dayofweek, date=df.tms.dt.date)
   .groupby("date").agg({"Transaction": "nunique", "dow": "first"})
   .reset_index()
   .groupby("dow")
   .Transaction
   .mean()
   .plot(kind='bar'))
# I am filtering out October and April months since the data only contain 
# few days from these.
(df.assign(month=df.tms.dt.month, date=df.tms.dt.date)
   .groupby("date").agg({"Transaction": "nunique", "month": "first"})
   .reset_index()
   .loc[lambda df: ~df.month.isin([10, 4]), :]
   .groupby("month")
   .Transaction
   .mean()
   .plot(kind='bar'))