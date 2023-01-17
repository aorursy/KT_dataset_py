#apps.csv: contains all the details of the applications on Google Play. There are 13 features that describe a given app.
# Read in dataset
import pandas as pd
apps = pd.read_csv("../input/apps.csv")

# Column names to check for duplication
column_names = ['App']
duplicates = apps.duplicated(subset = column_names, keep = False)
# Output duplicate values
apps[duplicates].sort_values(by = 'App')

# Drop duplicates
#apps = apps_with_duplicates.drop_duplicates(inplace=True)

# Print the total number of apps
print('Total number of apps in the dataset = ', 9659)

# Print a concise summary of apps dataframe
display(apps.info())

#print first five rows
display(apps.head())
# Have a look at a random sample of n rows
n = 5
apps.sample(n)
# List of characters to remove
chars_to_remove = ['+','$']
# List of column names to clean
cols_to_clean = ['Installs','Price']

# Replace each character with an empty string
apps["Installs"] = apps["Installs"].str.replace("+", "")
apps["Installs"] = apps["Installs"].str.replace(",", "")
apps["Price"] = apps["Price"].str.replace("$", "")
# Convert col to numeric
apps['Installs'] = pd.to_numeric(apps['Installs']) 
apps['Price'] = pd.to_numeric(apps['Price']) 
apps.info()
# Print the total number of unique categories
categories=apps['Category'].unique()
print(categories)
num_categories = len(apps['Category'].unique())
print('Number of categories = ', num_categories)
# Count the number of apps in each 'Category' and sort them in descending order
num_apps_in_categ = apps['Category'].value_counts().sort_values(ascending = False)
print(num_apps_in_categ)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 4))
num_apps_in_categ.plot(kind='bar')
plt.xlabel("categories",fontsize=14)
plt.ylabel("NO.off apps")
plt.tick_params(labelsize=12)
plt.show()

# Average rating of apps
avg_app_rating = apps['Rating'].mean()
print('Average app rating = ', avg_app_rating)

apps['Rating'].plot(kind='hist',bins=25)
plt.axvline(avg_app_rating, color='green', linewidth=2 , linestyle='--')
plt.xlabel("Rating")
plt.show()
# Filter rows where both Rating and Size values are not null
apps_Non_Null = apps[(~apps['Rating'].isnull()) & (~apps['Size'].isnull())]
# Plot size vs. rating
sns.scatterplot(x ='Size', y ='Rating', data = apps_Non_Null,alpha=0.2)
plt.show()

# Subset apps whose 'Type' is 'Paid'
paid_apps = apps_Non_Null[apps_Non_Null['Type'] == 'Paid']

# Plot price vs. rating
sns.scatterplot(x = 'Price', y ='Rating', data = paid_apps)
plt.show()

# Select a few popular app categories
popular_app_cats = apps[apps.Category.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY',
                                            'MEDICAL', 'TOOLS', 'FINANCE',
                                            'LIFESTYLE','BUSINESS'])]

sns.scatterplot(x="Price",y="Category",data= popular_app_cats)
plt.show()
# Apps whose Price is greater than 200
apps_above_200=popular_app_cats[popular_app_cats['Price']>200]
Category_App_Price = apps_above_200[['Category', 'App', 'Price']]
Category_App_Price
# Select apps priced below $100
apps_under_100 = popular_app_cats[popular_app_cats['Price']<100]

# Examine price vs category with the authentic apps
sns.scatterplot(x="Price",y="Category",data= apps_under_100)
plt.show()
# Data for paid apps
paid=apps[apps['Type'] == 'Paid']
# Data for free apps
free=apps[apps['Type'] == 'Free']


fig, ax=plt.subplots()
ax.boxplot([paid['Installs'],free['Installs']])
ax.set_yscale("log")
ax.set_xticklabels(["paid", "free"])
ax.set_ylabel("Installs")
plt.show()
# Load user_reviews.csv
reviews_df = pd.read_csv('../input/user_reviews.csv')

# Join and merge the two dataframe
merged_df = pd.merge(apps, reviews_df, on = 'App', how = "inner")

# Drop NA values from Sentiment and Translated_Review columns
merged_df = merged_df.dropna(subset=['Sentiment', 'Translated_Review'])

merged_df.info()
# User review sentiment polarity for paid vs. free apps
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11, 8)
ax = sns.boxplot(x = 'Type', y = 'Sentiment_Polarity', data = merged_df)
ax.set_title('Sentiment Polarity Distribution')

