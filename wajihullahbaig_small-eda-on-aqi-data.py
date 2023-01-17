import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
inputData = pd.read_csv(r"../input/AQM-dataset-updated.csv");
# Any results you write to the current directory are saved as output.
print(inputData.dtypes)
print(inputData.columns)
print("Data shape:",inputData.shape)
print(inputData.head())
print(inputData.describe())
print(inputData.info())
# Check for any nulls
print(inputData.isnull().sum())

# Drop unwanted stuff
inputData = inputData.loc[:, ~inputData.columns.str.contains('^Unnamed')]
inputData = inputData.iloc[1:]

print ("***************************************")
print ("VISUALIZATIONS IN DATA SET")
print ("***************************************")

fig = plt.figure(figsize = (10,10))
ax = fig.gca()
sns.heatmap(inputData.corr(), annot=True, fmt=".2f")
plt.title("Correlation",fontsize=5)
plt.show()

fig = plt.figure(figsize = (10,10))
ax = fig.gca()
inputData.hist(ax=ax)
plt.title("Histograms",fontsize =10)
plt.show()

fig = plt.figure(figsize = (15,15))
ax = fig.gca()
inputData.plot.area(ax=ax)
plt.title("Air quality indicators",fontsize =15)
plt.show()


fig = plt.figure(figsize = (10,10))
ax = fig.gca()
sns.lineplot(x="Date", y="CO_AQI", data=inputData,err_style="bars", ci=68)
plt.title("Carbon Monoxide - Datewise",fontsize =10)
plt.show()

fig = plt.figure(figsize = (10,10))
ax = fig.gca()
sns.lineplot(x="Date", y="Humidity", data=inputData,err_style="bars", ci=68)
plt.title("Humidity - Datewise",fontsize =10)
plt.show()


fig = plt.figure(figsize = (10,10))
ax = fig.gca()
sns.lineplot(x="Date", y="Temperature", data=inputData,err_style="bars", ci=68)
plt.title("Temperature - Datewise",fontsize =10)
plt.show()

