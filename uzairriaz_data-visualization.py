import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
food = pd.read_csv("../input/en.openfoodfacts.org.products.tsv",sep="\t")
foodAlt = food[['product_name','countries_en','energy_100g','main_category_en','proteins_100g']]
foodAlt = food[food.countries_en.notnull() & food.main_category_en.notnull()]

plt.figure(figsize=(5, 20))
food.isnull().mean(axis=0).plot.barh()
plt.title("Horizantal Bar Plot")
plt.show()

energy = foodAlt.energy_100g[food.countries_en == "United States"]
proteins = foodAlt.proteins_100g[food.countries_en == "United States"]
plt.scatter(energy,proteins)
plt.ylabel("Proteins")
plt.xlabel("Energy")
plt.title("Energy vs Proteins Scatter Plot for US")
plt.show()
