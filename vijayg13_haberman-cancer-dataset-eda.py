# Loading Nessesary Modules

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



'''downloaded dataset from https://drive.google.com/open?id=1o1I9PLyjqGgs0eOylK-2srXM2ZH3mIVb'''



#Load haberman.csv into a pandas dataFrame.

haberman = pd.read_csv("../input/haberman.csv")
# A breif look on dataframe

print("Haberman Dataset Head\n")

print(haberman.head(10))
# Dataset Description

print("Haberman Dataset Describe\n")

print(haberman.describe())
print("Haberman Dataset Info\n")

print(haberman.info())
# how many data-points and features?

print (haberman.shape)
# Check for any Null values

print(haberman.isnull().values.any())
print(list(haberman['status'].unique()))
haberman['status'] = haberman['status'].map({1:"survived", 2:"not_survived"})

haberman['status'] = haberman['status'].astype('category')

print(haberman.head(10))
print(haberman.info())
print("Number of rows: " + str(haberman.shape[0]))

print("Number of columns: " + str(haberman.shape[1]))

print("Columns: " + ", ".join(haberman.columns))

print("*"*100)

print("Target variable distribution")

print(haberman.iloc[:,-1].value_counts())

print("*"*100)

print(haberman.iloc[:,-1].value_counts(normalize = True))

print("*"*100)

print(haberman.describe())
# cite : https://www.kaggle.com/gokulkarthik/haberman-s-survival-exploratory-data-analysis

for idx, feature in enumerate(list(haberman.columns)[:-1]):

    fg = sns.FacetGrid(haberman, hue='status', height=5)

    fg.map(sns.distplot, feature).add_legend()

    plt.suptitle(str("Distribution Plot for "+feature), y=1.05, fontsize=18)

    plt.show()

    
plt.figure(figsize=(20,5))

for idx, feature in enumerate(list(haberman.columns)[:-1]):

    plt.subplot(1, 3, idx+1)

    print("\n*************** "+str(feature)+" ****************")

    counts, bin_edges = np.histogram(haberman[feature], bins=10, density=True)

    pdf = counts/sum(counts)

    print("PDF: {}".format(pdf))

    print("Bin Edges: {}".format(bin_edges))

    cdf = np.cumsum(pdf)

    print("CDF: {}".format(cdf))

    plt.plot(bin_edges[1:], pdf, label = "pdf")

    plt.plot(bin_edges[1:], cdf, label = "cdf")

    plt.xlabel(feature)

    plt.legend(loc='upper left')

    plt.title(str(feature+"'s pdf-cdf plot"))

    plt.grid(True)

    plt.suptitle("PDF-CDF Plottings", y=1.05, fontsize=20)
#Mean, Std-deviation, 



yes = haberman.loc[haberman["status"] == "survived"];

no = haberman.loc[haberman["status"] == "not_survived"];



for idx, feature in enumerate(list(haberman.columns)[:-1]):

    print(str(feature))

    print("Mean of ", str(feature)," of survived class : ",np.mean(yes[feature]))

    print("Mean of ", str(feature)," of not survived class : ",np.mean(no[feature]))

    print("Std Dev. of ", str(feature)," of survived class : ",np.std(yes[feature]))

    print("Std Dev. of ", str(feature)," of not survived class : ",np.std(no[feature]))

    print("\n")

#Median, Quantiles, Percentiles, IQR.

for idx, feature in enumerate(tuple(haberman.columns)[:-1]):

    print("*"*10,str(feature),"*"*10)

    print("\nMedians:")

    print(np.median(yes[feature]))

    print(np.median(no[feature]))



    print("\nQuantiles:")

    print(np.percentile(yes[feature],np.arange(0, 100, 25)))

    print(np.percentile(no[feature],np.arange(0, 100, 25)))



    print("\n90th Percentiles:")

    print(np.percentile(yes[feature],90))

    print(np.percentile(no[feature],90))



    from statsmodels import robust

    print ("\nMedian Absolute Deviation")

    print(robust.mad(yes[feature]))

    print(robust.mad(no[feature]))

    print("\n")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, feature in enumerate(tuple(haberman.columns)[:-1]):

    sns.boxplot(x='status',y=feature, hue="status", data=haberman, ax=axes[idx]).set_title(str("status-"+feature))

plt.suptitle("Box Plots",y=1.0, fontsize=18)

plt.show()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, feature in enumerate(tuple(haberman.columns)[:-1]):

    sns.violinplot(x='status',y=feature, hue="status", data=haberman, ax=axes[idx]).set_title(str("status-"+feature))

plt.suptitle("Violin Plots",y=1.05, fontsize=18)

plt.show()
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for idx, feature in enumerate(list(haberman.columns)[:-1]):

    for i, f in enumerate(list(haberman.columns)[:-1]):

        sns.scatterplot(x=feature, y=f, hue='status', ax=axes[idx][i], data = haberman).set_title(str(feature+" vs. "+f))

plt.suptitle("2-d Scatter Plots", fontsize=18)

plt.show()

plt.close();

sns.set()

sns.set_style("whitegrid");

sns.pairplot(haberman, hue="status", height=6)

plt.suptitle("PairPlots", y=1.05, fontsize = 18, color='black')

plt.show()
#2D Density plot, contors-plot

for idx, feature in enumerate(list(haberman.columns)[:-1]):

    for i, f in enumerate(list(haberman.columns)[:-1]):

        if(idx>=i):

            continue

        sns.jointplot(x=feature, y=f, data=yes, kind="kde");

        plt.suptitle(str("contour plot for "+feature+" & "+f), y = 1.05, fontsize=18)

        plt.show()