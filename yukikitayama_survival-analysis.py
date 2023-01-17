import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import statistics

from sklearn.impute import SimpleImputer

from lifelines import KaplanMeierFitter, CoxPHFitter

from lifelines.statistics import logrank_test

from scipy import stats
df = pd.read_csv("../input/echocardiogram.csv")

df.head()
print(df.isnull().sum())

print(df.shape)
imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')

COLUMNS = ['age', 'pericardialeffusion', 'fractionalshortening', 'epss', 'lvdd', 'wallmotion-score']

X = imp_mean.fit_transform(df[COLUMNS])

df_X = pd.DataFrame(X,

                    columns = COLUMNS)

df_X.shape
COLUMNS_keep = ['survival', 'alive']

df_keep = df[COLUMNS_keep]

df_keep.shape
df = pd.concat([df_keep, df_X], axis = 1)

df = df.dropna() 

print(df.isnull().sum())

print(df.shape)
sns.pairplot(df)
df.loc[df.alive == 1, 'dead'] = 0

df.loc[df.alive == 0, 'dead'] = 1

df.groupby('dead').count()
kmf = KaplanMeierFitter()

T = df['survival']

E = df['dead']

kmf.fit(T, event_observed = E)

kmf.plot()

plt.title("Kaplan Meier estimates")

plt.xlabel("Month after heart attack")

plt.ylabel("Survival")

plt.show()
print(statistics.median(df['age']))

print(statistics.median(df['wallmotion-score']))
age_group = df['age'] < statistics.median(df['age'])

ax = plt.subplot(111)

kmf.fit(T[age_group], event_observed = E[age_group], label = 'below 62')

kmf.plot(ax = ax)

kmf.fit(T[~age_group], event_observed = E[~age_group], label = 'above 62')

kmf.plot(ax = ax)

plt.title("Kaplan Meier estimates by age group")

plt.xlabel("Month after heart attack")

plt.ylabel("Survival")
score_group = df['wallmotion-score'] < statistics.median(df['wallmotion-score'])

ax = plt.subplot(111)

kmf.fit(T[score_group], event_observed = E[score_group], label = 'Low score')

kmf.plot(ax = ax)

kmf.fit(T[~score_group], event_observed = E[~score_group], label = 'High score')

kmf.plot(ax = ax)

plt.title("Kaplan Meier estimamtes by wallmotion-score group")

plt.xlabel("Month after heart attack")

plt.ylabel("Survival")
month_cut = 24

df.loc[(df.dead == 1) & (df.survival <= month_cut), 'censored'] = 1

df.loc[(df.dead == 1) & (df.survival > month_cut), 'censored'] = 0

df.loc[df.dead == 0, 'censored'] = 0

E_v2 = df['censored']



T_low = T[score_group]

T_high = T[~score_group]

E_low = E_v2[score_group]

E_high = E_v2[~score_group]



results = logrank_test(T_low, T_high, event_observed_A = E_low, event_observed_B = E_high)

results.print_summary()
cph = CoxPHFitter()

df_score_group = pd.DataFrame(score_group)

df_model = df[['survival', 'censored', 'age']]

df_model = pd.concat([df_model, df_score_group], axis = 1)

cph.fit(df_model, 'survival', 'censored')

cph.print_summary()
# p-value of Log-likelihood ratio test

round(stats.chi2.sf(10.68, 2),4)