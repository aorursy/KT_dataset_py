import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns
df = pd.read_csv("../input/Pokemon.csv")
df.head()
df = df.replace(np.nan, "Blank")
type_combination = df.groupby(["Type 1", "Type 2"]).size().reset_index(name="count")
type_combination
pd.crosstab(df["Type 1"], df["Type 2"])
plt.figure(figsize=(12, 12))
sns.heatmap(pd.crosstab(df["Type 1"], df["Type 2"]),
            cmap="YlGnBu", annot=True, cbar=False)
