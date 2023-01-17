import numpy as np

import pandas as pd

%matplotlib inline

pd.set_option('display.max_rows', 20)





df = pd.read_csv("../input/Car_sales.csv")

df
print("Keskmine võimsus(hobujõududes):", df["Horsepower"].mean())

print("Maksimaalne võimsus(hobujõududes):", df["Horsepower"].max())

print("Minimaalne võimsus(hobujõududes):", df["Horsepower"].min())



df.plot.scatter("Horsepower", "Engine_size", alpha=0.7);

df.Fuel_efficiency.plot.hist(bins=19, grid=False, rwidth=0.90); 
df.groupby("Manufacturer")["Price_in_thousands"].mean().sort_values(ascending=False)*1000