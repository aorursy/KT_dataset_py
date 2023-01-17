import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



divorce = pd.read_csv("../input/divorce-data/divorce_comp.csv")

marriage = pd.read_csv("../input/marriage/marriage_comp.csv", index_col = 0)
sns.catplot(x="age_compatibility", data=divorce, kind = 'count')
divorce.groupby('age_compatibility')['w_luna'].count()/len(divorce)*100
sns.catplot(x="age_compatibility", data=divorce, kind = 'count')
marriage.groupby('age_compatibility')['w_luna'].count()/len(marriage)*100
divorce2 = pd.read_csv("../input/divorce-data/divorce_lasted_comp.csv")
sns.relplot(x="age_compatibility", y="days_lasted", data=divorce2, kind="line", ci="sd")
from scipy.stats.stats import spearmanr

r,p_value = spearmanr(divorce2.age_compatibility, divorce2.days_lasted)

print('% .3f'%r,'% .3f'%p_value)