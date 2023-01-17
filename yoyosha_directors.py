import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pylab as plt
main = pd.read_csv("../input/directors-analysis/Main_v.02_CSV.csv", encoding='latin1')
main.head(12)
IQR_Oscars = pd.read_csv("../input/directors-analysis/IQR  Oscars CSV.csv", encoding='latin1')
IQR_Oscars.head()
gross = pd.read_csv("../input/directors-analysis/Gross_v.02_CSV.csv")
gross.head()
rating = pd.read_csv("../input/directors-analysis/Ratings_v.02_CSV.csv", encoding='latin1')
rating.head()
sns.set(rc={'figure.figsize':(16,9)})
with sns.color_palette("hls", 8):
    ax = sns.boxplot(x="Weighted", y="Name", data=main).set_title("IMDb", fontsize = 20)
    ax = sns.stripplot(x="Weighted", y="Name", data=main)
with sns.color_palette("hls", 8):
    ax = sns.boxplot(x="Metascore", y="Name", data=main).set_title("Metascore", fontsize = 20)
    ax = sns.stripplot(x="Metascore", y="Name", data=main)
sns.set(rc={'figure.figsize':(16,12)})
ax = sns.boxplot(x="Rating", y="Name", hue="Type", data=rating, palette="hls").set_title("IMDb & Metascore", fontsize = 20)
plt.legend(fontsize='12')
sns.set(rc={'figure.figsize':(16,12)})
with sns.color_palette("hls", 4):
    ax = sns.barplot(x="IQR", y="Name", hue="Audience", data=IQR_Oscars).set_title("Interquartile range of the 12 Directors", fontsize = 20)
    plt.legend(fontsize='12')
sns.set(rc={'figure.figsize':(12,8)})
ax = sns.barplot(x="Name", y="Avg. Net Gross", data=gross).set_title("Net Gross of the 12 Directors", fontsize = 20)
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(14,8)})
ax = sns.barplot(x="Name", y="Count", hue="Academy Awards", data=IQR_Oscars, palette=sns.mpl_palette("Set2")).set_title("Number of Oscars by Directors", fontsize = 20)
plt.xticks(rotation=75)
plt.legend(fontsize='12')