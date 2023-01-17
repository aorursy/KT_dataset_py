import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
apps_df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv', sep=',',header=None)

# rename columns
apps_df.rename(columns={0: 'app', 2: 'rating', 4: 'size'}, inplace=True)

# select app, rating, size
apps_df = apps_df[['app', 'rating', 'size']]

apps_df.size
# turn objects into float objects
apps_df['rating'] = pd.to_numeric(apps_df['rating'], errors='coerce')

# remove rows with nan from ratings
apps_df = apps_df[apps_df['rating'].notna()]

apps_df.head()
# remove rows that have 'varies with device' for size
apps_df = apps_df[~apps_df['size'].str.match("Varies with device")]
apps_df.head()
def extractRealNumberFromStr(x):
    number = x
    
    # remove plusses if present
    number = number.replace('+', '')
    
    # remove commas as we're working with periods
    number = number.replace(',', '')
    
    if 'k' in x:
        number = x.split('k')[0]
        
        number = float(number)
        
        # replace 1000 symbol with 1000
        number *= pow(10,3)
    elif 'M' in x:
        number = x.split('M')[0]
        
        number = float(number)
        
        # replace million symbol with million
        number *= pow(10,6)
    return float(number)
# extract real numbers from size column
apps_df['size'] = apps_df['size'].map(lambda x: extractRealNumberFromStr(x))
apps_df['size']
# remove app ratings that have typos
apps_df = apps_df[apps_df['rating'] <= 5.0]
# group by rating and size
count_series = apps_df.groupby(['rating','size']).size()
grouped_df = count_series.to_frame(name = 'count').reset_index()
grouped_df.head()
# scatterplot
plt.figure(figsize=(20,20))
plt.ylim(0.0, 1.05*pow(10,8))
plt.title('Frequency of app size per rating', fontsize=24)
plt.ylabel('Size (in bytes)', fontsize=16)
plt.xlabel('Rating', fontsize=16)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.1e}')) # 2 decimal places
plt.scatter(grouped_df['rating'], grouped_df['size'], grouped_df['count']*10, alpha=1)
plt.show()
# mean app size per rating
mean_df = apps_df.groupby(['rating'])['size'].mean().reset_index()
mean_df.head()
# std dev of mean app size per rating
std_df = apps_df.groupby(['rating'])['size'].std().reset_index()
std_df.head()

plt.figure(figsize=(10,10))
plt.title('Mean app size per rating', fontsize=24)
plt.ylabel('Size (in bytes)', fontsize=16)
plt.xlabel('Rating', fontsize=16)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.1e}')) # 2 decimal places
plt.scatter(mean_df['rating'], mean_df['size'], alpha=1)
plt.show()
# merge std dev with mean
merged_df = pd.DataFrame(mean_df['size'].tolist(), columns = ['mean']) 
merged_df['std'] = std_df['size']
merged_df.head()
grouped_df.boxplot(by='rating', column =['size'], grid = False, figsize=(20,20))
plt.figure(figsize=(20,20))
plt.title('Distribution of app size per rating', fontsize=24)
plt.ylabel('Size (in bytes)', fontsize=16)
plt.xlabel('Rating', fontsize=16)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:.1e}')) # 2 decimal places
plt.boxplot(merged_df, labels=mean_df['rating'])
plt.show()
print("Mean correlation")
print("Pearson: {}, Spearman: {}, Kendall: {}".format(
    mean_df['size'].corr(mean_df['rating'], method="pearson"),
    mean_df['size'].corr(mean_df['rating'], method="spearman"),
    mean_df['size'].corr(mean_df['rating'], method="kendall")))