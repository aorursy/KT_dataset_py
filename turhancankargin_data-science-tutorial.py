import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
cwurData = pd.read_csv('/kaggle/input/world-university-rankings/cwurData.csv')

print(cwurData.columns)

cwurData.head()
shanghaiData = pd.read_csv('/kaggle/input/world-university-rankings/shanghaiData.csv')

print(shanghaiData.columns)

shanghaiData.head()
timesData = pd.read_csv('/kaggle/input/world-university-rankings/timesData.csv')

print(timesData.columns)

timesData.head()
f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(cwurData.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
f,ax = plt.subplots(figsize=(11, 11))

sns.heatmap(shanghaiData.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(timesData.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

timesData.teaching.plot(kind = 'line', color = 'g',label = 'Teaching',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

timesData.research.plot(color = 'r',label = 'Research',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

timesData.plot(kind='scatter', x='teaching', y='research',alpha = 0.5,color = 'red')

plt.xlabel('teaching')              # label = name of label

plt.ylabel('research')

plt.title('Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

timesData.research.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
timesData.info()
def bar_plot(dataset, variable):

    """

        input: dataset ex:timesData, variable example: "country" or "university_name"

        output: bar plot & value count

    """

    # get feature

    var = dataset[variable]

    # count number of categorical variable(value/sample)

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))
category1 = ["country", "international", "university_name", "world_rank"]

for c in category1:

    bar_plot(timesData,c)
def plot_hist(dataset,variable):

    plt.figure(figsize = (9,3))

    plt.hist(dataset[variable], bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar = ["teaching", "research", "citations", "student_staff_ratio"]

for n in numericVar:

    plot_hist(timesData,n)
# 1 - Filtering Pandas data frame

x = shanghaiData['national_rank']=="1"

shanghaiData[x]
# 1 - Filtering Pandas data frame

x = cwurData['citations']<5

cwurData[x]
# 2 - Filtering pandas with logical_and

cwurData[np.logical_and(cwurData['quality_of_faculty']<20, cwurData['publications']<15 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

shanghaiData[(shanghaiData['total_score']>70) & (shanghaiData['alumni']>50)]
# lets classify universities whether they have high or low teaching.

threshold = sum(timesData.teaching) / len(timesData.teaching)

timesData["ratio"] = ["high" if i > threshold else "low" for i in timesData.teaching]

timesData.loc[:10,["ratio","teaching"]] # we will learn loc more detailed later
# Lets look at does cwurData data have nan value

# As you can see there are 2200 entries. However broad_impact has 2000 non-null object so it has 200 null object.

cwurData.info()
# Lets check broad_impact

cwurData["broad_impact"].value_counts(dropna =False)

# As you can see, there are 200 NAN value
# Lets drop nan values

cwurData["broad_impact"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data

cwurData["broad_impact"].value_counts(dropna =False)
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # detect outlier and their indeces

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        # store indeces

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    

    return multiple_outliers
timesData.loc[detect_outliers(timesData,["research","teaching","citations"])]
timesData.boxplot(column="research",by = "year")

plt.show()