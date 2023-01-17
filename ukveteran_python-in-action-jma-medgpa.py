import matplotlib.pyplot as plt

import pandas as pd





dat = pd.read_csv('../input/gpa-and-medical-school-admission/MedGPA.csv')

dat.head()
plt.plot(dat.GPA)

plt.ylabel('GPA')

plt.show()
plt.plot(dat.MCAT)

plt.ylabel('MCAT')

plt.show()
import seaborn as sns

sns.lmplot( x="GPA", y="MCAT", data=dat, fit_reg=False, hue='Sex', legend=False)

plt.legend(loc='lower right')
sns.pairplot(dat, kind="reg")

plt.show()
sns.pairplot(dat, kind="scatter", hue="Sex", palette="Set2")

plt.show()

 

sns.pairplot(dat, kind="scatter", hue="Sex", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))

plt.show()