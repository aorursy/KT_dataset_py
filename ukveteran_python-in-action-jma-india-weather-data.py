import matplotlib.pyplot as plt

import pandas as pd





dat = pd.read_csv('../input/weather-data-in-india-from-1901-to-2017/Weather Data in India from 1901 to 2017.csv')

dat.head()
plt.plot(dat.JAN)

plt.ylabel('JAN')

plt.show()
plt.plot(dat.FEB)

plt.ylabel('FEB')

plt.show()
import seaborn as sns

sns.lmplot( x="YEAR", y="JAN", data=dat, fit_reg=False)

sns.lmplot( x="YEAR", y="FEB", data=dat, fit_reg=False)

sns.lmplot( x="YEAR", y="MAR", data=dat, fit_reg=False)

sns.lmplot( x="YEAR", y="APR", data=dat, fit_reg=False)

sns.lmplot( x="YEAR", y="MAY", data=dat, fit_reg=False)

sns.lmplot( x="YEAR", y="JUN", data=dat, fit_reg=False)
sns.pairplot(dat, kind="reg")

plt.show()
sns.pairplot(dat, kind="scatter", palette="Set2")

plt.show()

 

sns.pairplot(dat, kind="scatter", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))

plt.show()