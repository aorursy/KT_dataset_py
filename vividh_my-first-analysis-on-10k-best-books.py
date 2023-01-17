%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy import stats

import seaborn as sns



#Make plots larger

plt.rcParams['figure.figsize'] = (10,6)
books = pd.read_csv("C:/Users/vivid/Info_7390/Assignment1/books.csv", sep=',')

books.head()
books1 = books;

books1["original_publication_year"] = books["original_publication_year"].fillna(0.0).astype(int)

books1.head(10)
unique_language = books1.language_code.unique()

print(unique_language)
books1.dropna(thresh=2)
books1.isnull().sum()
books1 = books1.loc[books1['original_publication_year'] > 1300]

books1 = books1.loc[books1['books_count'] < 100]

books1 = books1.loc[books1['ratings_count'] < 100000]

books1 = books1.loc[books1['average_rating'] < 200000]

books1 = books1.loc[books1['ratings_1'] < 5000]

books1 = books1.loc[books1['ratings_2'] < 12000]

books1 = books1.loc[books1['ratings_3'] < 30000]

books1 = books1.loc[books1['ratings_4'] < 50000]

books1 = books1.loc[books1['ratings_5'] < 60000]
books1['original_publication_year'].fillna(0).astype(int)

books1.tail()
books1.groupby(["original_publication_year", "language_code"]).count()
pd.crosstab(books1.original_publication_year, [books1.language_code, books1.average_rating], rownames=['original_publication_year'], colnames=['language_code', 'average_rating'])
books1.sort_values(by='original_publication_year').head()
sns.distplot(books1['original_publication_year'])

# Below ploting shows that maximum books are published during years 1950 - 2017. Only few percentage of books were published in

# year 1300 - 1950.
plt.hist(books1['original_publication_year'])
sns.distplot(books1['average_rating'])

# Displays that maximum average rating comes around between 4.0 to 4.5
sns.distplot(books1['books_count'])

# It shows that maximum books from our 9000 datasets are having book count ranging from 20-40 and it decreases further.
sns.distplot(books1['ratings_count'])

#We got to know that maximum number of books got ratings from less than 20000 audience. Only few % books got ratings count around

#80000 t0 100000 or more.
sns.distplot(books1['ratings_1'])

#Shows that less than 5000 people reponse include rating 1 that too max group belong to 0 to 1000 range.
sns.distplot(books1['ratings_2'])

#Shows that around 12000 people gave rating 2, among them max group belong to range of 0-2000.
ax = sns.distplot(books1['ratings_3'])

#Shows that less than 30000 people reponse include rating 3 that too max group belong to 0 to 7000 range.
sns.distplot(books1['ratings_4'])

##Shows that around 50000 people gave rating 4, among them max group belong to range of 100-15000.
sns.distplot(books1['ratings_5'])

#Displays that maximum 60000 people gave rating 5, among them majority group range from 1000-15000.
books1.books_count.describe()
books1.original_publication_year.describe()
books1.average_rating.describe()
books1.ratings_count.describe()
books1.ratings_1.describe()
books1.ratings_2.describe()
books1.ratings_3.describe()
books1.ratings_4.describe()
books1.ratings_5.describe()
sns.boxplot(x=books1["original_publication_year"])

# Here publishing year cannot be considered as outliers as we are considering books from different years.
sns.boxplot(x=books1["ratings_count"])

#Maximum count belong around 15000-30000 range so outside that range everything is considered as outliers.
sns.boxplot(x=books1["ratings_1"])

#Maximum count belong around 500 range so everything outside that two bars range is considered as outliers. But actual 

#anomalies can be considered the range after 4000.
sns.boxplot(x=books1["ratings_2"])

#Maximum count belong around 2000 range so everything outside that two bars range is considered as outliers. But actual 

#anomalies can be considered the range after 10000
sns.boxplot(x=books1["ratings_3"])

#Maximum count belong around 5000 range so everything outside that two bars range is considered as outliers. But actual 

#anomalies can be considered the range after 27000
sns.boxplot(x=books1["ratings_4"])

#Maximum count belong around 10000 range so everything outside that two bars range is considered as outliers. But actual 

#anomalies can be considered the range after 42000
sns.boxplot(x=books1["ratings_5"])

#Maximum count belong around 10000 range so everything outside that two bars range is considered as outliers. But actual 

#anomalies can be considered the range after 55000
g = sns.jointplot(x="average_rating", y="ratings_count", data=books1, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Average Rating$", "$Ratings Count$")



#Average maximum rating range between 3.5 to 4.5 and maximum rating count is looking very dense ranging from 500 to 40000.
cmap = sns.diverging_palette(0, 255, sep=1, n=256, as_cmap=True)



correlations = books1[['original_publication_year','average_rating', 'ratings_count']].corr()

print (correlations)

sns.heatmap(correlations, cmap=cmap)



# Heat map shows that there is relatibity between all three variables and count is given below, it ranges from 1.00 to -1.00
sns.boxplot(x="average_rating", y="language_code", data=books1)
sns.pairplot(books1[['original_publication_year','average_rating','ratings_count']])
sns.regplot(x="ratings_count", y="average_rating", data=books1)
iris = sns.load_dataset("iris")

sns.pairplot(iris, hue="species")