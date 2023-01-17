import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats as stats

import matplotlib.pyplot as plt # data visualization
%matplotlib inline
plt.rc('figure', figsize=(10.0, 4.0))

import os
print(os.listdir("../input"))
heroes = pd.read_csv('../input/heroes.csv') # Import csv file
heroes = heroes[heroes['First appearance'] < 2020] # Filter
heroes.head() # Show first n rows (5 by default)
heroes.shape # Show DataFrame dimensions
heroes.sort_values(by='Weight', ascending=False).head() # Sort
heroes[['Name', 'First appearance']][heroes['First appearance'] < 1950].head() # Selezione condizionata
years = heroes['First appearance'].dropna() # Select column without null values
years.hist(bins=70, range=[1925, 2025]) # Show histogram (freuances) with limited x range
plt.xticks(np.arange(1925, 2026, step=10)) # Set ticks from(inclusive) to(exclusive) with step 
plt.xlabel('Anno prima apparizione')
plt.ylabel('Numero eroi')
plt.title("Numero di prime apparizioni per anno")
plt.show()
# L'hist (quasi!) si può fare anche come grafico a barre della frequenza
first_app_freq = pd.value_counts(heroes['First appearance'])
plt.bar(first_app_freq.index, first_app_freq.get_values()) # Valori x, y
plt.xlim((1925, 2025)) # Limiti sulle x
plt.show()
publisher_freq = pd.crosstab(index=heroes['Publisher'], columns=['Abs. freqence'], colnames=[''])
publisher_rel_freq = pd.crosstab(index=heroes['Publisher'], columns=['Rel. freqence'], colnames=[''], normalize=True)
publisher_percent_freq = (publisher_rel_freq.apply(lambda p: np.round(p, 4)*100).astype(str).apply(lambda s: s + '%'))
publisher_margin_freq = pd.crosstab(index=heroes['Intelligence'], columns=heroes['Gender'], margins=True)

# A volte i dati sono troppo diversi e vanno divisi in fasce
pd.crosstab(index=pd.cut(heroes['Weight'], bins=[30, 50, 80, 100, 200, 500, 1000]), columns=[heroes['Gender']])

# Grafico frequenza cumulativa
first_app_freq_cumulate = (pd.crosstab(index=heroes['First appearance'], columns=['Cumulate freq.'], colnames=['']).cumsum())
first_app_freq_cumulate.plot(marker='o', legend=False)
plt.show()
# Grafico delle frequenze
heroes['Publisher'].value_counts().plot.bar() # Ordina sulle y
plt.show()
publisher_freq.plot.bar() # Ordina sulle x
plt.show()
# Confrontare le frequenze relative sullo stesso grafico ha molto più senso
male_strength_freq = (pd.crosstab(index=heroes.loc[heroes['Gender']=='M','Strength'], columns='Abs. freq.', normalize=True).loc[:, 'Abs. freq.'])
female_strength_freq = (pd.crosstab(index=heroes.loc[heroes['Gender']=='F','Strength'], columns='Abs. freq.', normalize=True).loc[:, 'Abs. freq.'])
male_strength_freq.plot(marker='o', color='blue', legend=False)
female_strength_freq.plot(marker='o', color='pink', legend=False)
plt.show()
# Altri grafici:
heroes['Publisher'].value_counts().plot.pie() # A torta
plt.axis('equal')
plt.show()
plt.vlines(first_app_freq.index, 0, first_app_freq.get_values())
plt.plot(first_app_freq.index, first_app_freq.get_values(), 'o')
plt.show()
heroes[heroes['Gender']=='M'].plot.scatter('Height', 'Weight')

# Retta posizionata a mano
trend = lambda x: -1200 + x * 7
x_range = [170, 300]
line, = plt.plot(x_range, list(map(trend, x_range)), color='black')
line.set_dashes([3, 2])
line.set_linewidth(2)
plt.show()
years.describe() # mean, std, min, quartiles ...
years.quantile(.15)
years.plot.box(vert=False, whis='range') # range include gli outliers nel baffo
plt.show()
# Grafico Q-Q per cercare una dist comune
marvel = heroes.loc[(heroes['Publisher']=='Marvel Comics') & (heroes['Height'].between(150, 200))]
dc = heroes.loc[(heroes['Publisher']=='DC Comics') & (heroes['Height'].between(150, 200))]
marvel_sample = marvel['Height'].sample(50)
dc_sample = dc['Height'].sample(50)
n = float(50)
plt.plot(marvel_sample.quantile(np.arange(n)/n), dc_sample.quantile(np.arange(n)/n), 'o')
plt.show()
eye_color = heroes[pd.notnull(heroes['Eye color'])]['Eye color']

def gini(series): return 1 - sum(series.value_counts(normalize=True).map(lambda f: f**2))
print('gini: ' + str(gini(eye_color)))

def entropy(series): return sum(series.value_counts(normalize=True).map(lambda f: -f * np.log2(f)))
print('entropy: ' + str(entropy(eye_color)))
heights = heroes[heroes['Height'].between(120, 240)]['Height']
heights.plot.hist(bins=50, normed=True)

h = np.arange(140, 220)
p_mean = np.mean(heights)
p_std = np.std(heights)

norm = stats.norm.pdf(h, p_mean, p_std) #Probability density function
plt.plot(h, norm)
# Cumulative Distribution Function di una normale std.
h = np.arange(-5, 5, step=0.1)
plt.plot(h, stats.norm.pdf(h, 0, 1))
plt.plot(h, stats.norm.cdf(h, 0, 1))
p = .7
rv = stats.bernoulli(p)
x = np.arange(0, 2)
plt.vlines(x, 0, rv.pmf(x), label='pmf') # Probability Mass Function
plt.plot(x, stats.bernoulli.pmf(x, p), 'o')
y = np.linspace(-1, 2, 1000)
plt.step(y, stats.bernoulli.cdf(y, p), label='cdf')
plt.xlim((-1, 2))
plt.legend()
plt.show()
# Uniforme discreta
dist = stats.randint(1, 5)
x = np.arange(1, 5)
plt.vlines(x, 0, dist.pmf(x), label='pmf')
plt.plot(x, dist.pmf(x), 'o')
y = np.linspace(0, 5, 1000)
plt.step(y, dist.cdf(y), label='cdf')
plt.legend()
plt.show()
# Espnenziale
dist = stats.expon()
x = np.linspace(0, 5, 1000)
plt.plot(x, dist.pdf(x), label='pmf')
plt.plot(x, dist.cdf(x), label='cdf')
plt.legend()
plt.show()
