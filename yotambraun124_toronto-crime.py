import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

df =pd.read_csv("/kaggle/input/toronto-crime-rate-per-neighbourhood/Neighbourhood_Crime_Rates_(Boundary_File)_.csv")
df.head()
len(df)
df.index
print(df.keys())
df.Neighbourhood.unique()
df_group_by_Neighbourhood = df.groupby(["Neighbourhood","Population","Assault_2014"])["Assault_2019"].mean()
df_group_by_Neighbourhood
df["Assault_per_pop19"] = df.Assault_2019/ df.Population
df["Assault_per_pop14"] = df.Assault_2014/ df.Population
df["diff19-14"]=df["Assault_per_pop19"]-df["Assault_per_pop14"]
df["Assault_per_pop19"].head()
df_group_by_Neighbourhood = df.groupby(["Neighbourhood","Assault_per_pop14"])["Assault_per_pop19"].mean()
df_group_by_Neighbourhood
df_group_by_Neighbourhood_diff = df.groupby(["Neighbourhood","Population"])["diff19-14"].mean()
df_group_by_Neighbourhood_diff