import pandas as pd
df = pd.read_csv("../input/Health_AnimalBites.csv")
df.describe()
df.info()
import matplotlib.pyplot as plt
df["vaccination_yrs"].plot.hist()

plt.title("Years since the last vaccination")

plt.show()