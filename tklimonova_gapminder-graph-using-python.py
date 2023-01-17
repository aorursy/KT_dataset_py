import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np
gapminder_filepath = "../input/gapminder - gapminder.csv"

gapminder_data = pd.read_csv(gapminder_filepath, index_col='country')

gapminder_data.head()
plt.plot(gapminder_data.gdp_cap, gapminder_data.life_exp)

plt.show()
plt.scatter(gapminder_data.gdp_cap, gapminder_data.life_exp)

#A correlation will become clear, when you display the GDP per capita on a logarithmic scale

plt.xscale('log')

plt.show()
plt.hist(gapminder_data.life_exp)

plt.show()
#with more bins for clear visibility

plt.hist(gapminder_data.life_exp, bins=20)

plt.show()
plt.scatter(gapminder_data.gdp_cap, gapminder_data.life_exp)

plt.xscale('log')

plt.xlabel('GDP per Capita [in USD]')

plt.ylabel('Life Expectancy [in years]')

plt.title('World Development in 2007')

plt.show()
plt.scatter(gapminder_data.gdp_cap, gapminder_data.life_exp)

plt.xscale('log')

plt.xlabel('GDP per Capita [in USD]')

plt.ylabel('Life Expectancy [in years]')

plt.title('World Development in 2007')

plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])

plt.show()
# Store population as a numpy array: np_pop

np_pop = np.array(gapminder_data.population)

np_pop2 = np_pop*2

#Use seaborn scatterplot for better customization

sns.scatterplot(gapminder_data['gdp_cap'], gapminder_data['life_exp'], hue = gapminder_data['continent'], size = np_pop2, sizes=(20,400))

plt.grid(True)

plt.xscale('log')

plt.xlabel('GDP per Capita [in USD]')

plt.ylabel('Life Expectancy [in years]')

plt.title('World Development in 2007')

plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])

plt.show()
# Increase the graph size

plt.figure(dpi=150)

# Store population as a numpy array: np_pop

np_pop = np.array(gapminder_data.population)

np_pop2 = np_pop*2

#Let's delete the annoying legend

sns.scatterplot(gapminder_data['gdp_cap'], gapminder_data['life_exp'], hue = gapminder_data['continent'], size=np_pop2, sizes=(20,400), legend = False)

plt.grid(True)

plt.xscale('log')

plt.xlabel('GDP per Capita [in USD]', fontsize = 14)

plt.ylabel('Life Expectancy [in years]', fontsize = 14)

plt.title('World Development in 2007', fontsize = 20)

plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])

plt.show()
# Increase the graph size

plt.figure(dpi=150)

# Store population as a numpy array: np_pop

np_pop = np.array(gapminder_data.population)

np_pop2 = np_pop*2

#Let's change the opacity 

sns.scatterplot(gapminder_data['gdp_cap'], gapminder_data['life_exp'], hue = gapminder_data['continent'], legend = False, 

                size=np_pop2, sizes=(20,500), alpha = 0.8)

plt.grid(True)

plt.xscale('log')

plt.xlabel('GDP per Capita [in USD]', fontsize = 14)

plt.ylabel('Life Expectancy [in years]', fontsize = 14)

plt.title('World Development in 2007', fontsize = 20)

plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])

#Add description to the biggest countries

plt.text(1550, 67, 'India')

plt.text(5650, 75, 'China')

plt.show()