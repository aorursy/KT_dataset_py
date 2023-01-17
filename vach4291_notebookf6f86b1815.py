# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
HR_df=pd.read_csv("../input/HR_comma_sep.csv")

HR_df.head()
HR_df['sales'].value_counts()
HR_df['salary'].value_counts()
HR_df.info()
HR_df.plot(kind="scatter", x="satisfaction_level", y="last_evaluation")
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

sns.pairplot(HR_df.drop("sales", axis=1), hue="salary", size=3)
sns.boxplot(x="salary", y="satisfaction_level", data=HR_df)
sns.violinplot(x="salary", y="satisfaction_level", data=HR_df, saze=6)
sns.violinplot(x="sales", y="satisfaction_level", data=HR_df, saze=6)
HR_df["satisfaction_level"].plot.hist()
HR_df["last_evaluation"].plot.hist()
HR_df.corr()
corr = HR_df.corr()



sns.heatmap(corr,annot=True,cmap="terrain",linecolor="red", robust=False, square=False)



HR_df.describe().T