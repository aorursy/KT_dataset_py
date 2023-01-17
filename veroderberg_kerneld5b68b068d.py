import pandas as pd

pd.set_option('mode.chained_assignment', None)

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
file = "../input/leave-final/22-week-leave.xlsx"

df = pd.read_excel(file, header=None, usecols = list(range(8)))

#print(df.info())
columns = ["Date", "Location", "Start", "End", "Duration", "Topic", "Platform"]

df.columns = columns

df.dropna(inplace=True, how="all")

df.reset_index(inplace=True)

#print(df.info())
df.Platform = df.Platform.astype("category")

df.Topic = df.Topic.astype("category")

df.Location = df.Location.astype("category")

#print(df.info())
df.Start = df.Date + pd.to_timedelta(df.Start.astype(str))

df.End = df.Date + pd.to_timedelta(df.End.astype(str))
df.set_index(df.Start, inplace=True, drop=True)

#df.head()
df = df.drop(["index", "Date"], axis=1)

#df.head()
df.Duration = df.End - df.Start
df["Hours"] = pd.to_numeric(df.Duration.dt.total_seconds()/3600.0)

#print(df.Hours)
hours_by_topic = df.groupby("Topic").sum()

hours_by_topic.loc["python"] += 51 #Ajustement pour heures de formation précédant le congé

hours_by_topic.loc["SQL"] += 11 #Ajustement pour heures de formation précédant le congé

hours_by_topic.loc["data cleaning/EDA/datetime"] += 4 #Ajustement pour heures de formation précédant le congé

hours_by_topic.loc["env. management (bash, conda, pipenv)"] += 10 #Ajustement pour heures de formation précédant le congé

hours_by_topic = hours_by_topic.sort_values("Hours", ascending=False)

hours_by_topic



total_hours_start_dec = hours_by_topic.Hours.sum()

print("Nombre total d'heures d'étude, incluant les heures avant le début du congé : " + str(total_hours_start_dec))
sns.set_palette("Paired")

plt.style.use("seaborn-darkgrid")

plt.figure(figsize=(10,5))

sns.barplot(hours_by_topic.index, hours_by_topic.Hours, order=hours_by_topic.index)

plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

plt.ylabel("Heures")

plt.title("Sujets étudiés")

plt.tight_layout()

plt.savefig("sujets.png")
day_topic = df.pivot(index="Start", columns="Topic", values="Hours").resample("D").sum()

week_topic = df.pivot(index="Start", columns="Topic", values="Hours").resample("W").sum()

#day_topic
by_platform = df.groupby("Platform").sum()

by_platform
total_hours = by_platform.Hours.sum()

print("Nombre total d'heures depuis le début du congé : " + str(total_hours))
plt.pie(by_platform["Hours"], autopct="%1d%%", labels=by_platform.index)

#plt.savefig("plateformes.png")

plt.show()

platform_df = df.pivot(index="Start", columns="Platform", values="Hours")

platform_day = platform_df.resample("D").sum()

platform_week = platform_df.resample("W").sum()

platform_month = platform_df.resample("M").sum()

print("Nombre de semaines de congé : " + str(len(platform_week)))

#platform_week
def create_stacked_bar_chart(df_pivot, y_label):

    if "Week" in str(df_pivot.index.freq):

        periodicity = "Week"

    elif "Day" in str(df_pivot.index.freq):

        periodicity = "Day"

    elif "Month" in str(df_pivot.index.freq):

        periodicity = "Month"

    else:

        periodicity = "xxxx"

    values_list = []

    for column in df_pivot.columns:

        values_list.append(df_pivot[column])

    bottom_array = [0] * len(df_pivot)

    plt.figure(figsize=(16,6))

    for i in range(len(values_list)):

        plt.bar(range(len(df_pivot)), values_list[i], bottom=bottom_array)

        bottom_array += values_list[i]

    labels = [df_pivot.index[i].strftime("%Y-%m-%d") if bottom_array[i] > 0 else "" for i in range(len(df_pivot))]

    plt.axis([0 - np.sqrt(len(df_pivot)*0.16), len(df_pivot)+ np.sqrt(len(df_pivot)*0.16), 0, max(bottom_array)*1.05])

    plt.legend(labels=df_pivot.columns, loc="best")

    plt.xticks(range(len(df_pivot)), labels=labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=10)

    plt.ylabel(y_label)

    plt.title(y_label + " by " + df_pivot.columns.name + " by " + periodicity, fontsize=20)

    plt.savefig(periodicity + "_" + df_pivot.columns.name + df_pivot.index[-1].strftime("%Y-%m-%d") + ".png")

#    print(max(bottom_array))

#    print(len(df_pivot))

    plt.show()



create_stacked_bar_chart(platform_week, "Hours")    
platform_week
plt.figure(figsize=(16,4))

for column in platform_week.columns:

    plt.plot(range(len(platform_week)), platform_week[column])

plt.xticks(range(len(platform_week)), platform_week.index, rotation=45, ha="right", rotation_mode="anchor", fontsize=10)

plt.legend()

plt.show()
#plt.figure(figsize=(10,4))

#plt.stackplot(platform_week.index, platform_week.CodeCademy, platform_week.Coursera, platform_week.DataCamp, platform_week.Jupyter, platform_week.Khan, platform_week["Plotly Studio"], platform_week["intranet/Web"], platform_week.rencontres)

#plt.legend(platform_week.columns)

#plt.xticks(range(22), platform_week.index, rotation=45, ha="right", rotation_mode="anchor", fontsize=10)

#plt.show()
most_studied = hours_by_topic[0:9]

most_studied.index



most_studied_week = week_topic[most_studied.index]

most_studied_week.loc["2019-11-03 00:00:00"] = [0,0,0,0,0,0,0,0,0]

most_studied_week.index = pd.to_datetime(most_studied_week.index)
plt.figure(figsize=(16,4))

for column in most_studied_week.columns:

    plt.plot(range(len(most_studied_week)), most_studied_week[column])

plt.xticks(range(len(most_studied_week)), most_studied_week.index.strftime("%Y-%m-%d"), rotation=45, ha="right", rotation_mode="anchor", fontsize=10)

plt.axis([1, 26, 0, 23])

plt.xlabel("Week Ending on")

plt.ylabel("Hours by Subject")

plt.title("Time Spent on Most Studied Subjects")

plt.legend()

plt.show()
palettes = ["Spectral", "Blues", "cubehelix", "inferno", "viridis"] #"hsv", "husl","Paired", "magma",



for palette in palettes:

    plt.figure(figsize=(16,4))

    plt.stackplot(range(len(most_studied_week)), [most_studied_week[column] for column in most_studied_week.columns], colors=sns.color_palette(palette, 9))

    plt.xticks(range(len(most_studied_week)), most_studied_week.index.strftime("%Y-%m-%d") ,rotation=45, ha="right", rotation_mode="anchor", fontsize=10)

    plt.axis([1, 26, 0, 32])

    plt.legend(most_studied_week.columns)

    #plt.xlabel("Week Ending on")

    #plt.ylabel("Total Hours Color-Coded by Subject")

    #plt.title("Time Spent on Most Studied Subjects")

    plt.savefig("Most_Studied_Subjects_" + palette + ".jpg")

    plt.show()
for palette in palettes:

    plt.figure(figsize=(16,4))

    plt.stackplot(range(len(week_topic)), [week_topic[column] for column in week_topic.columns], colors=sns.color_palette(palette, 16))

    plt.xticks(range(len(week_topic)), week_topic.index.strftime("%Y-%m-%d") ,rotation=45, ha="right", rotation_mode="anchor", fontsize=10)

    plt.axis([1, 28, 0, 36])

    plt.legend(week_topic.columns)

    plt.xlabel("Week Ending on")

    plt.ylabel("Total Hours Color-Coded by Subject")

    plt.title("Time Spent on all Subjects")

    plt.show()
week_topic