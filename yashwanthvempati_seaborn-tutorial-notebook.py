#Visualizing statistical relationships



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")



tips = pd.read_csv('../input/seaborn-tips-dataset/tips.csv')

sns.relplot(x="total_bill", y="tip", data=tips);
sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips);

sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker",

            data=tips);
sns.relplot(x="total_bill", y="tip", hue="smoker", style="time", data=tips);

sns.relplot(x="total_bill", y="tip", hue="size", data=tips);
sns.relplot(x="total_bill", y="tip", hue="size", palette="ch:r=-.5,l=.75", data=tips);
sns.relplot(x="total_bill", y="tip", size="size", data=tips);
sns.relplot(x="total_bill", y="tip", size="size", sizes=(15, 200), data=tips);
#Emphasizing continuity with line plots



df = pd.DataFrame(dict(time=np.arange(500),

                       value=np.random.randn(500).cumsum()))

g = sns.relplot(x="time", y="value", kind="line", data=df)

g.fig.autofmt_xdate()



df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])

sns.relplot(x="x", y="y", sort=False, kind="line", data=df);
#Plotting with categorical data

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)



sns.catplot(x="day", y="total_bill", data=tips);
sns.catplot(x="day", y="total_bill", jitter=False, data=tips);

sns.catplot(x="day", y="total_bill", kind="swarm", data=tips);

sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips);

sns.catplot(x="size", y="total_bill", kind="swarm",

            data=tips.query("size != 3"));
sns.catplot(x="smoker", y="tip", order=["No", "Yes"], data=tips);

sns.catplot(x="total_bill", y="day", hue="time", kind="swarm", data=tips);
#Distributions of observations within categories



sns.catplot(x="day", y="total_bill", kind="box", data=tips);

sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips);

tips["weekend"] = tips["day"].isin(["Sat", "Sun"])

sns.catplot(x="day", y="total_bill", hue="weekend",

            kind="box", dodge=False, data=tips);
#Violinplots



sns.catplot(x="total_bill", y="day", hue="time",

            kind="violin", data=tips);



sns.catplot(x="total_bill", y="day", hue="time",

            kind="violin", bw=.15, cut=0,

            data=tips);



sns.catplot(x="day", y="total_bill", hue="sex",

            kind="violin", split=True, data=tips);



sns.catplot(x="day", y="total_bill", hue="sex",

            kind="violin", inner="stick", split=True,

            palette="pastel", data=tips);



g = sns.catplot(x="day", y="total_bill", kind="violin", inner=None, data=tips)

sns.swarmplot(x="day", y="total_bill", color="k", size=3, data=tips, ax=g.ax);
#Statistical estimation within categories



import pandas as pd

import seaborn as sns

titanic = pd.read_csv("../input/titanic-data-set/titanic.csv")

sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=titanic);

#Point plots

sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="point", data=titanic);



sns.catplot(x="Pclass", y="Survived", hue="Sex",

            palette={"male": "g", "female": "m"},

            markers=["^", "o"], linestyles=["-", "--"],

            kind="point", data=titanic);
#Plotting univariate distributions



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

sns.set(color_codes=True)



x = np.random.normal(size=100)

sns.distplot(x);
sns.distplot(x, kde=False, rug=True);
sns.distplot(x, bins=20, kde=False, rug=True);
sns.distplot(x, hist=False, rug=True);