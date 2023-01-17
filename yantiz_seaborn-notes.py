# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
# Path of the file to read
spotify_filepath = "../input/spotify/datasets_116573_334386_spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)
# Print the first 5 rows of the data
spotify_data.head()
# Line chart showing daily global streams of each song 
sns.lineplot(data=spotify_data)
# Set the width and height of the figure
plt.figure(figsize=(14,6))

# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")

# Line chart showing daily global streams of each song 
sns.lineplot(data=spotify_data)
# Set the width and height of the figure
plt.figure(figsize=(14,6))

# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")

# Line chart showing daily global streams of 'Shape of You'
sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")

# Line chart showing daily global streams of 'Despacito'
sns.lineplot(data=spotify_data['Despacito'], label="Despacito")

# Add label for horizontal axis
plt.xlabel("Date")
# Path of the file to read
flight_filepath = "../input/flight-delays/flight_delays.csv"

# Read the file into a variable flight_data
flight_data = pd.read_csv(flight_filepath, index_col="Month")
# Print the data
flight_data
# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['NK'])
# or sns.barplot(y=flight_data.index, x=flight_data['NK'])

# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")
# Set the width and height of the figure
plt.figure(figsize=(14,7))

# Add title
plt.title("Average Arrival Delay for Each Airline, by Month")

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=flight_data, annot=True)

# Add label for horizontal axis
plt.xlabel("Airline")
insurance_filepath = "../input/insurance/insurance.csv"

# Read the file into a variable insurance_data
insurance_data = pd.read_csv(insurance_filepath)

insurance_data.head()
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
sns.swarmplot(x=insurance_data['smoker'], y=insurance_data['charges'])
# Path of the file to read
iris_filepath = "../input/iris/Iris.csv"

# Read the file into a variable iris_data
iris_data = pd.read_csv(iris_filepath, index_col="Id")

# Print the first 5 rows of the data
iris_data.head()
# Histogram 
sns.distplot(a=iris_data['PetalLengthCm'], kde=False)
# KDE plot 
sns.kdeplot(data=iris_data['PetalLengthCm'], shade=True)
# 2D KDE plot
sns.jointplot(x=iris_data['PetalLengthCm'], y=iris_data['SepalWidthCm'], kind="kde")
# break the dataset into three separate dataframes
iris_set_data = iris_data[iris_data['Species'] == 'Iris-setosa']
iris_ver_data = iris_data[iris_data['Species'] == 'Iris-versicolor']
iris_vir_data = iris_data[iris_data['Species'] == 'Iris-virginica']

# Print the first 5 rows of the Iris versicolor data
iris_ver_data.head()
# Histograms for each species
sns.distplot(a=iris_set_data['PetalLengthCm'], label="Iris-setosa", kde=False)
sns.distplot(a=iris_ver_data['PetalLengthCm'], label="Iris-versicolor", kde=False)
sns.distplot(a=iris_vir_data['PetalLengthCm'], label="Iris-virginica", kde=False)

# Add title
plt.title("Histogram of Petal Lengths, by Species")

# Force legend to appear
plt.legend()
# KDE plots for each species
sns.kdeplot(data=iris_set_data['PetalLengthCm'], label="Iris-setosa", shade=True)
sns.kdeplot(data=iris_ver_data['PetalLengthCm'], label="Iris-versicolor", shade=True)
sns.kdeplot(data=iris_vir_data['PetalLengthCm'], label="Iris-virginica", shade=True)

# Add title
plt.title("Distribution of Petal Lengths, by Species")
# Path of the file to read
spotify_filepath = "../input/spotify/datasets_116573_334386_spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)
# Change the style of the figure to the "dark" theme
sns.set_style("dark")

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)