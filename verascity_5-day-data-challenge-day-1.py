import pandas as pd
cereal_data = pd.read_csv("../input/cereal.csv")
print(cereal_data.shape)
print(cereal_data.describe())
print(cereal_data.head())
