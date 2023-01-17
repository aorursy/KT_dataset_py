%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
salaries = pd.read_csv("../input/Salaries.csv")
salaries = salaries[salaries['BasePay'] != 'Not Provided']
salaries["BasePay"] = salaries.BasePay.astype("float")
salaries["JobTitle"] = salaries.JobTitle.str.replace("\\d+|\b\w\b","").str.lower()
salaries.head(3)
for name, group in salaries.groupby("JobTitle"):
    if len(group) < 1000:
        continue
    plt.figure()
    plt.title(name)
    sns.distplot(group.Benefits.apply(float).dropna())
    plt.show()