import pandas as pd
import matplotlib.pyplot as plt
cereal_data = pd.read_csv("../input/cereal.csv")
print(cereal_data.info())
print(cereal_data[["name", "sugars"]])
plt.hist(cereal_data["sugars"], range=(0, 15))
plt.hist(cereal_data["sugars"], bins=12, range=(0, 15), edgecolor="black")
plt.title("Sugar Content of Popular Cereals")
plt.xlabel("Sugar in grams (per serving)")
plt.ylabel("Count")