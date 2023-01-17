path = '../input/'

filename = "Melbourne_housing_extra_data-18-08-2017.csv"



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import cm

import seaborn as sns

import scipy.stats as scs 

import statsmodels.api as sm



data = pd.read_csv(path + filename)
data.head()
#%% Exploration of the data



group_region = np.asarray(data.groupby('Regionname')['Lattitude','Longtitude'])

color=iter(cm.rainbow(np.linspace(0,1,len(group_region))))





plt.clf()

plt.grid(False)

for k in range(len(group_region)):

  c = next(color)

  plt.scatter(group_region[k][1]['Lattitude'],

              group_region[k][1]['Longtitude'],c=c,s = 10.0)

plt.title("Region of properties in the Melbourne area")

plt.xlabel("Lattitude")

plt.ylabel("Longtitude")

plt.show()



  

group_suburb = np.asarray(data.groupby('Suburb')['Lattitude','Longtitude'])

color=iter(cm.rainbow(np.linspace(0,1,len(group_region))))





plt.clf()

plt.grid(False)

color=iter(cm.rainbow(np.linspace(0,1,len(group_region))))

for k in range(len(group_region)):

  c = next(color)

  plt.scatter(group_region[k][1]['Lattitude'], 

              group_region[k][1]['Longtitude'],c=c,s=10.0)



plt.clf()

plt.grid(False)

color=iter(cm.rainbow(np.linspace(0,1,len(group_region))))

for k in range(len(group_region)):

  c = next(color)

  plt.scatter(group_region[k][1]['Lattitude'], 

            group_region[k][1]['Longtitude'],c=c,s = 10*group_region[k][1]['Price']/np.mean(group_region[k][1]['Price']))

plt.title("Price of properties in the Melbourne area")

plt.xlabel("Lattitude")

plt.ylabel("Longtitude")

plt.show()



plt.clf()

plt.grid(False)

color=iter(cm.rainbow(np.linspace(0,1,len(group_region))))

for k in range(len(group_region)):

  c = next(color)

  plt.scatter(group_region[k][1]['Lattitude'], 

            group_region[k][1]['Longtitude'],c=c,s = 5*group_region[k][1]['Landsize']/np.mean(group_region[k][1]['Landsize']))

plt.title("Landsize of properties in the Melbourne area")

plt.xlabel("Lattitude")

plt.ylabel("Longtitude")

plt.show()



plt.clf()

plt.grid(False)

color=iter(cm.rainbow(np.linspace(0,1,len(group_region))))

for k in range(len(group_region)):

  c = next(color)

  plt.scatter(group_region[k][1]['Lattitude'], 

            group_region[k][1]['Longtitude'],c=c,s = 5*group_region[k][1]['BuildingArea']/np.mean(group_region[k][1]['BuildingArea']))

plt.title("Building Area of properties in the Melbourne area")

plt.xlabel("Lattitude")

plt.ylabel("Longtitude")

plt.show()



#%% Exploration of the distribution of prices



# for all properties



plt.clf()

plt.grid(False)

plt.hist(data["Price"].dropna(), bins = 100)

plt.title("Price distribution of properties in the Melbourne area")

plt.xlabel("Price")

plt.ylabel("Count")

plt.show()



# splitting by region



plt.clf()

plt.grid(False)

color=iter(cm.rainbow(np.linspace(0,1,len(group_region))))

for k in range(len(group_region)):

  plt.clf()

  c = next(color)

  plt.hist(group_region[k][1]['Price'].dropna(),color=c,bins = 100)

  plt.title("Price of properties in the " +  group_region[k][0] + " area")

  plt.xlabel("Price")

  plt.ylabel("Count")

  plt.show()

  

plt.clf()

plt.grid(False)

color=iter(cm.rainbow(np.linspace(0,1,len(group_region))))

for k in range(len(group_region)):

  plt.clf()

  c = next(color)

  plt.hist(np.log(group_region[k][1]['Price'].dropna()),color=c,bins = 100)

  plt.title("Price of properties in the " +  group_region[k][0] + " area")

  plt.xlabel("Price")

  plt.ylabel("Count")

  plt.show()



## We notice that we have too little information about some regions

## suggesting we should drop them in further study

## other price distributions drop quickly



# splitting by house type



group_type = np.asarray(data.groupby(by = "Type"))

plt.clf()

plt.grid(False)

color=iter(cm.rainbow(np.linspace(0,1,len(group_type))))

for k in range(len(group_type)):

  plt.clf()

  c = next(color)

  plt.hist(group_type[k][1]['Price'].dropna(),color=c,bins = 100)

  plt.title("Price of properties of type " +  group_type[k][0])

  plt.xlabel("Price")

  plt.ylabel("Count")

  plt.show()
## We notice that we have too little information about some regions

## suggesting we should drop them in further study

## other price distributions drop quickly

## it seems like the lognormal distribution property holds well



## Note: those functions were taken from Yves Hilpisch's Python for Finance



def normality_tests(arr):

  print("Skew of data set %14.3f" % scs.skew(arr))

  print("Skew test p-value %14.3f" % scs.skewtest(arr)[1])

  print("Kurt of data set %14.3f" % scs.kurtosis(arr))

  print("Kurt test p-value %14.3f" % scs.kurtosistest(arr)[1]) 

  print("Norm test p-value %14.3f" % scs.normaltest(arr)[1])

  

def print_statistics(array):

  sta = scs.describe(array)

  print("%14s %15s" % ('statistic', 'value'))

  print(30 * "-")

  print ("%14s %15.5f" % ('size', sta[0]))

  print ("%14s %15.5f" % ('min', sta[1][0]) )

  print ("%14s %15.5f" % ('max', sta[1][1]))

  print ("%14s %15.5f" % ('mean', sta[2]))

  print ("%14s %15.5f" % ('std', np.sqrt(sta[3])))

  print ("%14s %15.5f" % ('skew', sta[4]))

  print ("%14s %15.5f" % ('kurtosis', sta[5]))



group_type = np.asarray(data.groupby(by = "Type"))

plt.clf()

plt.grid(False)

color=iter(cm.rainbow(np.linspace(0,1,len(group_type))))

for k in range(len(group_type)):

  plt.clf()

  c = next(color)

  plt.hist(np.log(group_type[k][1]['Price'].dropna()),color=c,bins = 100)

  plt.title("Price of properties of type " +  group_type[k][0])

  plt.xlabel("Price")

  plt.ylabel("Count")

  plt.show()

  

for k in range(len(group_type)):

  print("Normality test for type ",group_type[k][0])

  normality_tests(np.log(group_type[k][1]["Price"].dropna()))

  

for k in range(len(group_region)):

  print("Normality test for region ",group_region[k][0])

  normality_tests(np.log(group_region[k][1]["Price"].dropna()))
#% Using statistical models 

# Plot the PDF.



from scipy.stats import norm



group_type = np.asarray(data.groupby(by = "Type"))

group_type_filtered = np.copy(group_type)

plt.clf()

plt.grid(False)

color=iter(cm.rainbow(np.linspace(0,1,len(group_type))))

for k in range(len(group_type)):

  plt.clf()

  c = next(color)

  mu, std = norm.fit(np.log(group_type[k][1]["Price"].dropna()))

  plt.hist(np.log(group_type[k][1]["Price"].dropna()), bins=50, normed=True,color=c)

  xmin, xmax = plt.xlim()

  x = np.linspace(xmin, xmax, 100)

  p = norm.pdf(x, mu, std)

  plt.plot(x, p, 'k',alpha = 0.6, linewidth=2)

  title = "Fit results, Price of properties of type " +  group_type[k][0]+ " : mu = %.2f,  std = %.2f" % (mu, std)

  plt.title(title)

  plt.show()