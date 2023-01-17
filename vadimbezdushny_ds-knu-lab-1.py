import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
import scipy.stats as stats
import pylab

print(os.listdir("../input"))
df = pd.read_csv('../input/NBA_player_of_the_week.csv')
df = df['Weight']
df.unique()
df = df.apply(lambda s: float(s[:-2]) if 'kg' in s else int(s) * 0.453592) # unify kg and pounds
np.sort(df.unique())
df.describe()
df.median()
stats.variation(df)
stats.kurtosis(df)
stats.skew(df)
df.plot.box()
df.hist(bins=5)
df.hist(bins=10)
df.plot.density()
probplot = sm.ProbPlot(df.values, stats.t, fit=True)
_ = probplot.ppplot(line='45')
_ = probplot.qqplot(line='45')
