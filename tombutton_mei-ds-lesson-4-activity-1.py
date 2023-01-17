# import pandas
import pandas as pd

# import matplotlib
import matplotlib.pyplot as plt

#import the data and check by view the top rows
country_data = pd.read_csv('../input/meilds1/mei-lds-1.csv')
country_data.head()
# explore the data
# generate the box plot for the life expectancy at birth column
country_data.boxplot(column = ['life expectancy at birth'])
# the plt.show() command removed the additional output text generated from matplotlib - you can remove this
plt.show()
# draw a boxplot for life expectancy split up by region
country_data.boxplot(column = ['life expectancy at birth'],by='Sub region', vert=False,figsize=(12, 8))
plt.show()
# draw a boxplot for GDP split up by region

# draw a boxplots for any other fields that you think are relevant
# draw a histogram for country_data['life expectancy at birth']
country_data['life expectancy at birth'].plot.hist(bins=[40,50,60,70,80,90,100],density=1)
plt.show()
# draw a histogram for GDP
country_data['life expectancy at birth'].plot.density()
plt.show()

country_data['GDP per capita (US$)'].plot.density()
plt.show()
# draw a scatter diagram for life expectancy v GDP
country_data.plot.scatter(x='GDP per capita (US$)', y='life expectancy at birth')
plt.show()
# draw a scatter diagram for  birth rate per 1000 v GDP
# draw a hexagonal bin plot for life expectancy v GDP
country_data.plot.hexbin(x='GDP per capita (US$)', y='life expectancy at birth',gridsize=10)
plt.show()
# draw a hexagonal bin plot for birth rate per 1000 v GDP