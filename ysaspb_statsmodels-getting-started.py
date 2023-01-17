import statsmodels.api as sm

import pandas

from patsy import dmatrices
df = sm.datasets.get_rdataset("Guerry", "HistData").data
vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']

df = df[vars]

df[-5:]
df = df.dropna()

df[-5:]
y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')
y[:3]
X[:3]
mod = sm.OLS(y, X)    # Describe model

res = mod.fit()       # Fit model

print(res.summary())   # Summarize model
res.params
res.rsquared
dir(res)
sm.stats.linear_rainbow(res)
sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'],data=df, obs_labels=False)
