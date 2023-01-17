# make imports

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from scipy import stats
# read datasets (disasters)

carbon_df = pd.read_csv('../input/carbon-emissions/MER_T12_06.csv')

disasters_df = pd.read_csv('../input/natural-disaster-data/number-of-natural-disaster-events.csv')

econ_df = pd.read_csv('../input/natural-disaster-data/economic-damage-from-natural-disasters.csv')
# Look at the data for natural disasters

disasters_df.head()
# drop NaN values

disasters_df = disasters_df.drop(columns='Code')

econ_df = econ_df.drop(columns='Code')
disasters_df['Entity'].unique()
(disasters_df['Year'].min(), disasters_df['Year'].max())
def grab_decade(start_yr, y_c_data, interval=10):

    '''Return years and counts for only a specific interval length.'''

    end_yr = int(start_yr) + interval - 1

    years = y_c_data[(y_c_data['years'] <= end_yr) & (y_c_data['years'] >= start_yr)]

    return years



def compute_decade_mean(start_yr, y_c_data):

    '''Sum the number of total disasters over a given period of 10 years, returns the mean.'''

    years = grab_decade(start_yr, y_c_data)

    # compute and return the mean

    return years['counts'].sum() / 10
def compute_means(y_c_data):

    '''Returns a dict of all mean number of disasters that occurred for every decade, 1900-2010.'''

    # compute the amount of decades in our data

    start_yr, end_yr = y_c_data['years'].min(), y_c_data['years'].max()

    decades = (end_yr - start_yr) // 10

    # store all the means in a dict

    decade_means = dict()

    for i in range(start_yr, end_yr, 10):

        decade_means[f'{i}'] = compute_decade_mean(i, y_c_data)

    return decade_means



# Calling the function

ALL_DIS = 'All natural disasters'

COUNT = 'Number of reported natural disasters (reported disasters)'

counts = disasters_df[(disasters_df['Entity'] == ALL_DIS)][COUNT]  # just the counts of all natural disasters, all years

years = disasters_df[(disasters_df['Entity'] == ALL_DIS)]['Year']  # just the years

y_c_data = pd.DataFrame(data={

                        'years':years, 

                        'counts':counts})

means_by_decade = compute_means(y_c_data)
plt.plot(list(means_by_decade.keys()), list(means_by_decade.values()))

plt.xlabel('Decade Start Year')

plt.ylabel('Annual Mean Disaster Count')

plt.title('Change in Decade Mean for Natural Disasters, 1900-2010')

plt.show()
def compute_decade_median(start_yr, y_c_data):

    '''Return the median of total disasters over a given period of 10 years.'''

    years = grab_decade(start_yr, y_c_data)

    # compute and return the median

    return years['counts'].median()



def compute_medians(y_c_data):

    '''Returns a dict of all mean number of disasters that occurred for every decade, 1900-2010.'''

    # compute the amount of decades in our data

    start_yr, end_yr = y_c_data['years'].min(), y_c_data['years'].max()

    decades = (end_yr - start_yr) // 10

    # store all the medians in a dict

    decade_medians = dict()

    for i in range(start_yr, end_yr, 10):

        decade_medians[f'{i}'] = compute_decade_median(i, y_c_data)

    return decade_medians



medians_by_decade = compute_medians(y_c_data)
plt.plot(list(medians_by_decade.keys()), list(medians_by_decade.values()))

plt.xlabel('Decade Start Year')

plt.ylabel('Median Disaster Count')

plt.title('Change in Decade Median for Natural Disasters, 1900-2010')

plt.show()
counts = disasters_df[(disasters_df['Entity'] == 'All natural disasters') & (disasters_df['Year'] >= 2000) & (disasters_df['Year'] <= 2010)]['Number of reported natural disasters (reported disasters)']

plt.plot(list(range(2000, 2011)), counts)

plt.xlabel('Year')

plt.ylabel('Annual Mean Disaster Count')

plt.title('Change in Natural Disaster Count, 2000-2010')

plt.show()
# find all rows reporting "all natural disasters"

COUNT = 'Number of reported natural disasters (reported disasters)'

all_disasters = disasters_df[disasters_df['Entity'] == 'All natural disasters'][COUNT]

# sum them together, divide by their number

mean_disasters = np.sum(all_disasters) / len(all_disasters)

# print the mean

mean_disasters
count = 0

for num in all_disasters:

    if num > mean_disasters:

        count += 1

count
all_disasters_years_and_counts = disasters_df[(disasters_df['Entity'] == 'All natural disasters')]

years_2000_2018 = all_disasters_years_and_counts.tail(19)

count = 0

for num in years_2000_2018['Number of reported natural disasters (reported disasters)']:

    if num > mean_disasters:

        count += 1

        

percent_val = round((count/19) * 100, 2)  

print(f'{percent_val}%')  # have all these years surpassed the mean we calculated?
print(f'{round((19/42) * 100, 2)}%')
# slice the DataFrame by century

disasters_20th = disasters_df[(disasters_df['Entity'] == 'All natural disasters') & (disasters_df['Year'] <= 1999) & (disasters_df['Year'] >= 1900)]

disasters_21st = disasters_df[(disasters_df['Entity'] == 'All natural disasters') & (disasters_df['Year'] >= 2000) & (disasters_df['Year'] <= 2018)]



# find the mean annual number of disasters in the 20th century

mean_20th = disasters_20th[COUNT].values.mean()



# compute the percent of years in the 21st century which is greater than this value

percent_over = len(disasters_21st[disasters_21st[COUNT] > mean_20th]) / len(disasters_21st) * 100

print(f'{percent_over}%')
# find the total number of years with counts above the mean_20th

count_above_mean = len(all_disasters[all_disasters > mean_20th])

print(f'{round((18/count_above_mean) * 100, 2)}%')
# let's take another look at that data

all_disasters_years_and_counts
y_c_data
plt.plot(y_c_data['years'], y_c_data['counts'])

plt.title('All Natural Disasters Globally, From 1900-2018')

plt.ylabel('Total Count')

plt.xlabel('Year')

plt.show()
def probability_for_interval(start_year, end_year):

    # take the sum of all natural disasters that occurred 1900-2018

    sum_all = y_c_data['counts'].sum()

    # take the sum that happen over the interval

    yrs_in_range = y_c_data[(y_c_data['years'] < end_year) & (y_c_data['years'] > start_year)]

    sum_yrs = yrs_in_range['counts'].sum()

    # return the probability

    percent = round((sum_yrs/sum_all) * 100, 2)

    return percent

    

prob_20th = probability_for_interval(1900, 2000)

print(f'{prob_20th}%')
prob_21st = probability_for_interval(2000, 2018)

print(f'{prob_21st}%')
plt.pie([prob_20th, prob_21st], labels=['20th', '21st'])

plt.title('Relative Frequency of Natural Disasters in 20th & 21st Centuries')

plt.show()
def find_remove_outlier_iqr(disaster_counts):

    '''Remove the outliers from the dataset of annual total nautral disasters.'''

    # calculate interquartile range

    q25, q75 = np.percentile(disaster_counts, 25), np.percentile(disaster_counts, 75)

    iqr = q75 - q25

    print(f'This is the IQR: {iqr}')

    # calculate the outlier cutoff

    cut_off = iqr * 1.5

    lower, upper = q25 - cut_off, q75 + cut_off

    # identify outliers

    outliers = [x for x in disaster_counts if x < lower or x > upper]

    # remove outliers

    outliers_removed = [x for x in disaster_counts if x > lower and x < upper]

    return outliers



print(f'Number of outliers removed from the data: {len(find_remove_outlier_iqr(counts))}')
# show box plot

counts = all_disasters_years_and_counts['Number of reported natural disasters (reported disasters)']

plt.boxplot(counts)

plt.title("Box Plot of Annual Natural Disasters, 1900-2018")

plt.ylabel("Count of Natural Disasters")

plt.xlabel("Years 1900-2018")

plt.show()
carbon_df.head()

carbon_df['Description'].values
carbon_df.tail()
# store the annual emissions count in a dict

years_emissions = dict()

# just look at emissions from total electric output

carbon_total = carbon_df[carbon_df['Description'] == 'Total Energy Electric Power Sector CO2 Emissions']

# traverse through the years

for i in range(197300, 201700, 100):

    # find all the rows in the data for the year we're currently on

    year = carbon_total[(carbon_total['YYYYMM'] >= i) & (carbon_total['YYYYMM'] <= i + 12)]

    # sum the emissisons for that one year

    sum = 0.0

    for value in year['Value']:

        # handle the invalid values

        if value == 'Not Available':

            value = 0.0

        sum += float(value)

    # store it in the dict

    years_emissions[int(i/100)] = sum

# Voila! A dict of all years and their emissions counts, 1973-2016

print(years_emissions)

# One of the things to note in this data is that NaN values were replaced 0, but this is likely far from the

# true number of emissions made that month
plt.plot(list(years_emissions.keys()), list(years_emissions.values()))

plt.title('Annual Carbon Emissions from Electricity Generation, 1973-2016')

plt.xlabel('Year')

plt.ylabel('Million Metric Tons of Carbon Dioxide')

plt.show()
econ_df.head()
# combining datasets

df = disasters_df.rename(columns={'Number of reported natural disasters (reported disasters)': 'Disaster Count'})

df2 = econ_df.rename(columns={'Total economic damage from natural disasters (US$)':'Cost'})

df['Cost'] = df2['Cost']

df.head()

dollars = df[df['Entity'] == 'All natural disasters']['Cost']

plt.plot(years, dollars)

plt.title('Cost of Nautral Disasters Globally, 1900-2018')

plt.ylabel('Total Cost (USD)')

plt.xlabel('Year')

plt.show()
# Credit to the Seaborn Documentation for inspiring this cell: https://seaborn.pydata.org/examples/many_pairwise_correlations.html

sns.set(style="white")

# Compute the correlation matrix

corr = df.corr()

# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(8, 8))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Covariance Between Costs Against Counts')

plt.show()
def pearson_corr(x, y):

    '''Given two lists of numbers x and y, return the value of their Pearson correlation coefficient.'''

    x_mean = np.mean(x)

    y_mean = np.mean(y)

    num = [(i - x_mean)*(j - y_mean) for i,j in zip(x,y)]

    den_1 = [(i - x_mean)**2 for i in x]

    den_2 = [(j - y_mean)**2 for j in y]

    correlation_x_y = np.sum(num)/np.sqrt(np.sum(den_1))/np.sqrt(np.sum(den_2))

    return correlation_x_y



# get a lists of the counts and the costs

counts = df[(df['Entity'] == 'All natural disasters') & (df['Year'] <= 2018) & (df['Year'] >= 1900)]['Disaster Count']

costs = df[(df['Entity'] == 'All natural disasters') & (df['Year'] <= 2018) & (df['Year'] >= 1900)]['Cost']

corr_cost_count = pearson_corr(costs, counts)

print(f'Correlation between cost of damages and disaster count: {corr_cost_count}.')
# 1-sample t-test

# get a list of the costs of disasters for just the 21st century

costs = df[df['Entity'] == 'All natural disasters']['Cost'].values

costs_21 = df[(df['Entity'] == 'All natural disasters') & (df['Year'] <= 2018) & (df['Year'] >= 2000)]['Cost'].values



# calculate the mean cost annually due to disasters, for whole population (1900-2018)

pop_mean = costs.mean()



# run the test

t, p = stats.ttest_1samp(costs_21, pop_mean)



# see the results

print(f"The t-statistic is {t} and the p-value is {p}.")
import pandas as pd

economic_damage_from_natural_disasters = pd.read_csv("../input/natural-disaster-data/economic-damage-from-natural-disasters.csv")

number_of_natural_disaster_events = pd.read_csv("../input/natural-disaster-data/number-of-natural-disaster-events.csv")
import pandas as pd

MER_T12_06 = pd.read_csv("../input/carbon-emissions/MER_T12_06.csv")