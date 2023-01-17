import matplotlib.pyplot as plt

import pandas as pd





dat = pd.read_csv('../input/insurance/insurance.csv')

dat.head()
plt.plot(dat.bmi)

plt.ylabel('BMI')

plt.show()
plt.plot(dat.charges)

plt.ylabel('Charges')

plt.show()
import seaborn as sns

sns.lmplot( x="bmi", y="charges", data=dat, fit_reg=False, hue='sex', legend=False)

plt.legend(loc='upper left')
sns.pairplot(dat, kind="reg")

plt.show()
sns.pairplot(dat, kind="scatter", hue="sex", palette="Set2")

plt.show()

 

sns.pairplot(dat, kind="scatter", hue="sex", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))

plt.show()