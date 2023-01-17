# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/creditcard.csv")
data.columns.unique()
data.describe()
data.groupby("Class").describe()
pc_columns = ["V{:d}".format(i) for i in np.arange(1, 29)]
explained_variances = data[pc_columns].apply(np.var).to_frame()



explained_variances.columns=["variance"]



explained_variances_ratio = explained_variances / explained_variances.sum()

explained_variances_ratio.columns = ["Var Ratio"]



explained_variances_ratio_cumsum = explained_variances_ratio.cumsum()

explained_variances_ratio_cumsum.columns = ["Var Ratio Cumsum"]
import matplotlib.pyplot as  plt

plt.style.use("ggplot")



_, ax = plt.subplots(figsize=(12, 4))

explained_variances_ratio_cumsum.plot(kind="line", color="#ee7621", ax=ax, linestyle="-", marker="h")

explained_variances_ratio.plot(kind="bar", ax=ax, color="#00304e", alpha=0.8, rot=0, fontsize=7)

ax.set_title("Explained Variance Ratio of Principle Components", fontsize=10)

ax.set_ylim([0.0, 1.1])

    

for x, y in zip(np.arange(0, len(explained_variances_ratio_cumsum)), explained_variances_ratio_cumsum["Var Ratio Cumsum"]):

    ax.annotate("{:.1f}%".format(y * 100.0), xy=(x-0.25, y+0.03), fontsize=6)



for x, y in zip(np.arange(1, len(explained_variances_ratio)), explained_variances_ratio["Var Ratio"][1:]):

    ax.annotate("{:.1f}%".format(y * 100.0), xy=(x-0.2, y+0.02), fontsize=6)
import seaborn as sns



_ = sns.pairplot(

    data = pd.concat([

        data[data.Class == 0].sample(500),

        data[data.Class == 1]

    ]),

    vars=pc_columns[:5],

    hue="Class",

    diag_kind="kde"

)
