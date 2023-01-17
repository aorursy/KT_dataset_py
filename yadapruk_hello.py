%matplotlib inline
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn
seaborn.set(style='ticks')
df = pd.DataFrame.from_csv("../input/Speed Dating Data.csv", encoding="ISO-8859-1",index_col=False)
keep_cols = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1", "match", "expnum", "gender"] 
df = df[keep_cols]
#adding up all the matches for every 10 rows to find total number of actual matches for each ID. 
actual_matches = []
actual_matches = [(df.iloc[i:i+10-1].sum(axis=0)["match"])for i in range(0, len(df), 10)]
keep_cols = ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1", "match", "expnum", "gender"] 
df = df[keep_cols]
#adding up all the matches for every 10 rows to find total number of actual matches for each ID. 
#df['match']  = actual_matches
df = df.iloc[::10, :] 
df["match"] = actual
_matches

_genders= [0, 1]
df = pd.DataFrame({
    'Self-labeled level of sincerity': df["sinc3_1"], 
    'Predicted Number of Matches': df["match"],
    'Gender': df["gender"]
})

ax2 = seaborn.FacetGrid(data=df, hue='Gender', hue_order=_genders, aspect=1.61)
ax2.map(pyplot.scatter, 'Self-labeled level of sincerity','Predicted Number of Matches').add_legend()



