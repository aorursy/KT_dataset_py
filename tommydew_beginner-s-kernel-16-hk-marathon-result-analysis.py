import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/challenge.csv")

df.head()
df.isnull().sum()
df = df.dropna()
df.info()
df.describe()
df['Official Time'].apply(type).unique()
def time_conversion(x):

    time_split = [int(i) for i in x.split(":")]

    return time_split[0] + time_split[1]/60 + time_split[2]/3600



net_time_in_decimal = df['Net Time'].apply(time_conversion)



sns.distplot(net_time_in_decimal)

plt.show()
time_col = ['Official Time', 'Net Time', '10km Time', 'Half Way Time', '30km Time']

for i in time_col:

    df[i] = df[i].apply(time_conversion)
df['Category'].unique()
df['Gender'] = df['Category'].apply(lambda x:x[1])



sns.kdeplot(df[df['Gender'] == "M"]['Official Time'], label = 'Male')

sns.kdeplot(df[df['Gender'] == "F"]['Official Time'], label = 'Female')

plt.title("Runners' Results by Genders")

plt.legend()

plt.show()
Middle_Position = df[df['Gender'] == "F"]['Gender Position'].iloc[-1] / 2 #455

first_half_F = df[(df['Gender'] == "F") & (df['Gender Position'] < Middle_Position)]

last_half_F = df[(df['Gender'] == "F") & (df['Gender Position'] > Middle_Position)]



sns.distplot(first_half_F['Official Time'])

sns.distplot(last_half_F['Official Time'])

plt.show()
last_half_F.head()
Middle_Position = df[df['Gender'] == "M"]['Gender Position'].iloc[-1] / 2 #455

first_half_F = df[(df['Gender'] == "M") & (df['Gender Position'] < Middle_Position)]

last_half_F = df[(df['Gender'] == "M") & (df['Gender Position'] > Middle_Position)]



sns.distplot(first_half_F['Official Time'])

sns.distplot(last_half_F['Official Time'])

plt.show()
last_half_F.head()
df['Unknown Factor'] = df['Category'].apply(lambda x:x[-1])



unknowns= df['Unknown Factor'].unique()



plt.figure()

for i in unknowns:

    sns.kdeplot(df[df['Unknown Factor'] == i]['Official Time'], label = i)



plt.legend()

plt.title("Runners' Results by Unknown Factors")



plt.figure()

sns.barplot(x = 'Unknown Factor', y = 'Official Time', data = df, hue = 'Gender')

plt.title("Mean of results by Unknown Factor")

plt.legend(loc = 'center', bbox_to_anchor = (0.5, 0.95), ncol = 2)

plt.tight_layout()



plt.show()
plt.figure(figsize = (12, 8))

fig = sns.violinplot(y = 'Official Time', x = 'Unknown Factor', data = df, hue = 'Gender')



result_byUnknownCat = []

for p in fig.patches:

    fig.annotate(round(p.get_height(), 2), (p.get_x(), p.get_height() + 0.05), fontsize = 10)

    result_byUnknownCat.append(p.get_height())

    

plt.title("Official Time by Unknown Factors and Genders")

plt.show()
df[df['Unknown Factor'] == "I"]['Country '].unique()
print("Summation of overall positions of runners within the category")

for i in ["S", "1", "2"]:

    sum_of_Post = df[df['Unknown Factor'] == i]['Overall Position'].sum()

    print("{}:".format(i), sum_of_Post)
cat_S = df[df['Unknown Factor'] == "S"]['Country '].unique()



cat_1 = df[df['Unknown Factor'] == "1"]['Country '].unique()



cat_2 = df[df['Unknown Factor'] == "2"]['Country '].unique()



print("S: {}\n1: {}\n2: {}".format(cat_S, cat_1, cat_2))
print("Countries in common:\ncat1 in catS: {}\ncat2 in catS: {}\ncat1 in cat2: {}".format(

    sum([1 for i in cat_1 if i in cat_S]), sum([1 for i in cat_2 if i in cat_S]), 

    sum([1 for i in cat_1 if i in cat_2])))
PTable = pd.pivot_table(df, values = 'Official Time', index = ['Unknown Factor'], columns= ['Gender'])



PTable['Difference (F - M)'] = PTable['F'] - PTable['M']

PTable['Ratio (F / M)'] = PTable['F'] / PTable['M']

PTable.sort_values(by = 'M')
df['Country '] = df['Country '].replace({"Macau": "Macao SAR"})



countries = df['Country '].unique()

results_Bycountries = [[np.mean(df[(df['Country '] == i)]['Official Time'])] for i in countries]

results_dict = zip(countries, results_Bycountries)

results_sorted = sorted(results_dict, key = lambda x:x[1])



plt.figure(figsize = (12, 18))

sns.boxplot(y = 'Country ', x = 'Official Time', data = df, orient = 'h', order = [i[0] for i in results_sorted])

plt.show()
df['Sec Half Time'] = df['Official Time'] - df['Half Way Time']



plt.scatter(y = df['Half Way Time'], x = df['Sec Half Time'], alpha = 0.2)

plt.plot([1, 4], [1, 4], linestyle = '-', color = 'r')

plt.show()
df[df['Sec Half Time']  < df['10km Time'] * 1.7]
fig = plt.figure()

ax1 = fig.add_subplot(211)

ax2 = fig.add_subplot(212)



fig.suptitle("Hong Kong SAR Results Distribution")

sns.distplot(df[df['Country '] == 'Hong Kong SAR']['Official Time'], ax = ax1)



sns.distplot(df[(df['Country '] == 'Hong Kong SAR') & (df['Gender'] == "M")]['Official Time'], label = "M", ax = ax2)

sns.distplot(df[(df['Country '] == 'Hong Kong SAR') & (df['Gender'] == "F")]['Official Time'], label = "F", ax = ax2)



ax1.set_xlabel("Official Time")

ax2.set_xlabel("Official Time")

fig.align_xlabels()

plt.show()
fastCountry = ['Ethiopia', 'Kenya']

for i in fastCountry:

    fig = plt.figure()

    ax1 = fig.add_subplot(211)

    ax2 = fig.add_subplot(212)



    fig.suptitle("{} Results Distribution".format(i))

    sns.distplot(df[df['Country '] == i]['Official Time'], ax = ax1)



    sns.distplot(df[(df['Country '] == i) & (df['Gender'] == "M")]['Official Time'], label = "M", ax = ax2)

    sns.distplot(df[(df['Country '] == i) & (df['Gender'] == "F")]['Official Time'], label = "F", ax = ax2)



    ax1.set_xlabel("Official Time")

    ax2.set_xlabel("Official Time")

    fig.align_xlabels()
country = list(df["Country "].value_counts().iloc[:5].index)

for i in country:

    fig = plt.figure()

    ax1 = fig.add_subplot(211)

    ax2 = fig.add_subplot(212)

    

    fig.suptitle("{} Results Distribution".format(i))

    sns.kdeplot(df[df['Country '] == i]['Official Time'], ax = ax1)

    

    sns.distplot(df[(df['Country '] == i) & (df['Gender'] == "M")]['Official Time'], label = "M", ax = ax2)

    FemaleD = df[(df['Country '] == i) & (df['Gender'] == "F")]['Official Time']

    if len(FemaleD) > 1:

        sns.distplot(df[(df['Country '] == i) & (df['Gender'] == "F")]['Official Time'], label = "F", ax = ax2)

    plt.legend()

    ax1.set_xlabel("Official Time")

    ax2.set_xlabel("Official Time")

    fig.align_xlabels()

    plt.show()