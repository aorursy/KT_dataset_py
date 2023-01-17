import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def group_by(col):
    grouped = kobe.groupby(col)
    hits = grouped.shot_made_flag.sum()
    d = {'#hits': hits, '#no hits': grouped.shot_made_flag.count()-hits}
    df = pd.DataFrame(data=d)
    return df
    

kobe = pd.read_csv('../input/data.csv')
kobe = kobe[kobe.shot_made_flag >= 0]

print("By Season")
by_season = group_by('season')
print(by_season)
by_season.plot.barh(stacked=True, color=['b','r'])
plt.savefig('by_season.png')