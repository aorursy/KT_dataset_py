# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

#         print(os.path.join(dirname, filename))

        files.append(os.path.join(dirname, filename))

# print(files)

# Any results you write to the current directory are saved as output.
print(files[1])

#2019-novel-coronavirus-covid-19-2019-ncov-data-repository-recovered.csv

print(files[1])

file_name = "/kaggle/input/uncover/UNCOVER/johns_hopkins_csse/2019-novel-coronavirus-covid-19-2019-ncov-data-repository-recovered.csv"

d_f = pd.read_csv(file_name)

print(d_f.head())

dates = d_f["date"].unique()

# print(dates)

intial = dates[0]

final = dates[-1]

print("intial :", intial, ", final :", final)

# print(d_f.head())

def plot_recovery(d_f,intial, final):

    dic_recover_sum = {"region":[],"recovered":[]}

    rec = []

    reg = []

    c_region = d_f['country_region'].unique()

#     print(c_region)

    dates = d_f["date"].unique()

    for region in c_region:

        d_f = d_f[d_f["date"] == final]

        rec_r = d_f[d_f["country_region"] == region]["recovered"].sum()

#         print(d_f[d_f["country_region"] == "China"]["recovered"].sum())

        if rec_r > 5000:

            rec.append(rec_r)

            reg.append(region)

    dic_recover_sum["recovered"] = rec # Data between 2020-01-22 - 2020-03-31

    dic_recover_sum["region"] = reg

    df_rec = pd.DataFrame(dic_recover_sum)

    mean_recovery = df_rec["recovered"].mean() 

    median_recovery = df_rec["recovered"].median() 

    std_recovery = df_rec["recovered"].std() 

    cond1 = mean_recovery + 2*std_recovery

    outliner = df_rec[df_rec["recovered"]>cond1]

    print("outliner cond : ", cond1, ", outliner : ", outliner)

    plt.scatter(rec,reg, c=[i for i in range(len(reg))])

#     plt.show()

# for date in  dates:

plot_recovery(d_f, intial, final)

# plot_recovery(d_f, intial, intial)    

# plot_geo_plot() # need to work on
print(files[2])

file_name2 = "/kaggle/input/uncover/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-states.csv"

d_f = pd.read_csv(file_name2)

c_region = d_f["country_region"].unique()

print(d_f.head())

# total_cases = region_df["confirmed"].sum()

# death_cases = region_df["deaths"].sum()

# recovered_cases = region_df["recovered"].sum()

# active_cases = region_df["active"].sum()

# recovery_ratio = np.round(recovered_cases/total_cases,2)

# death_ratio = np.round(death_cases/total_cases,2)

# active_ratio = np.round(active_cases/total_cases,2)

region_frame = {}

total_cases = []

death_cases = []

recovered_cases = []

active_cases = []

recovery_ratio = []

death_ratio = []

active_ratio = []

for region in c_region:

    print(region)

    region_df = d_f[d_f["country_region"] == region]

    total_cases.append(region_df["confirmed"].sum())

    death_cases.append(region_df["deaths"].sum())

    recovered_cases.append(region_df["recovered"].sum())

    active = region_df["confirmed"].sum() - region_df["deaths"].sum() - region_df["recovered"].sum()

    active_cases.append(active)

    recovery_ratio.append(np.round(region_df["recovered"].sum()/region_df["confirmed"].sum(),2))

    death_ratio.append(np.round(region_df["deaths"].sum()/region_df["confirmed"].sum(),2))

    active_ratio.append(np.round(active/region_df["confirmed"].sum(),2))

region_frame["country"] = c_region

region_frame["total_cases"] = total_cases

region_frame["death_cases"] = death_cases

region_frame["recovered_cases"] = recovered_cases

region_frame["active_cases"] = active_cases

region_frame["recovery_ratio"] = recovery_ratio

region_frame["death_ratio"] = death_ratio

region_frame["active_ratio"] = active_ratio

region_frame = pd.DataFrame(region_frame)

print(region_frame)



def autolabel(rects,ax):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 4, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')



def plot_ratios(region_frame, ratio):

    

    labels = region_frame["country"]

#     men_means = [20, 34, 30, 35, 27]

#     women_means = [25, 32, 34, 20, 25]



    x = np.arange(len(labels))  # the label locations

    width = 0.35/2  # the width of the bars



    fig, ax = plt.subplots()

    if ratio == 0:

        rects1 = ax.bar(x - width/4, region_frame["total_cases"], width, label='total_cases')

        rects2 = ax.bar(x - width/2, region_frame["death_cases"], width, label='death_cases')

        rects3 = ax.bar(x + width/2, region_frame["recovered_cases"], width, label='recovered_cases')

        rects4 = ax.bar(x + width/4, region_frame["active_cases"], width, label='active_cases')

    elif ratio == 1:

        rects1 = ax.bar(x - width/4, region_frame["active_ratio"], width, label='active_ratio')

        rects2 = ax.bar(x - width/2, region_frame["death_ratio"], width, label='death_ratio')

        rects3 = ax.bar(x + width/2, region_frame["recovery_ratio"], width, label='recovery_ratio')

#     rects2 = ax.bar(x + width/2, women_means, width, label='Women')



    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax.set_ylabel('Cases')

    ax.set_title('field of cases ')

    ax.set_xticks(x)

    ax.set_xticklabels(labels)

    ax.legend()

    

#     autolabel(rects1)

#     autolabel(rects2)

#     autolabel(rects3)

#     autolabel(rects4)



    fig.tight_layout()



    plt.show()

plot_ratios(region_frame,0)

plot_ratios(region_frame,1)

dropus = region_frame[region_frame["country"]!="US"]

dropchina = dropus[region_frame["country"]!="China"]

plot_ratios(dropchina,0)

plot_ratios(dropchina,1)









# region_frame.plot(y=c_region)