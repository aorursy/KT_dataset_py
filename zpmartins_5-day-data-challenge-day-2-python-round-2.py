# Following:

# http://mailchi.mp/3355c773a1e2/data-challenge-day-1-read-in-and-summarize-a-csv-file-2576405
import matplotlib.pyplot as plt

import pandas as pd 
nutrition = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")
nutrition.describe(include="all").transpose()
# List Column Names

nutrition.columns
sodium_column = nutrition[" Sodium (mg)"]
# Draw Histagram

plt.hist(sodium_column, bins=6, edgecolor="black")

plt.title("Sodium in Starbucks Menu Items")

plt.xlabel("Sodium in milligrams")

plt.ylabel("Count")
nutrition.hist(column=" Sodium (mg)", bins=6)
nutrition.plot(kind='hist', subplots=True, title='Starbucks Menu Items', figsize=(20,20), sharey=False, sharex=False, )