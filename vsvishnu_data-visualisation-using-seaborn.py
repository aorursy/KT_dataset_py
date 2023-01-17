import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.plotting.register_matplotlib_converters()

%matplotlib inline
spotify_data = pd.read_csv("../input/csvfiles/spotify.csv", index_col = "Date")
spotify_data = pd.read_csv("../input/csvfiles/spotify.csv", index_col = "Date",parse_dates = True)
spotify_data.head()
plt.figure(figsize = (14, 6))

plt.title("Daily Global Streams of Popular Songs in 2017-18")



sns.lineplot(data = spotify_data)
sns.set_style('white')



plt.figure(figsize = (14, 6))

sns.lineplot(data = spotify_data, palette='husl') 
ign_data = pd.read_csv('../input/datafiles/ign_scores.csv', index_col = "Platform")

ign_data.head()
plt.figure(figsize = (30, 6))

plt.title("Average rating of RPG games, by Platform")



sns.barplot(x = ign_data.index, y = ign_data["RPG"])
plt.figure(figsize = (14,6))

plt.title("Average Rating of Each Genre, by Platform")



sns.heatmap(data = ign_data, annot = True)
insurance_data = pd.read_csv("../input/datafiles/insurance.csv")

insurance_data.head()
plt.figure(figsize = (14,6))



sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
plt.figure(figsize = (14, 6))

sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
plt.figure(figsize = (14 , 6))

sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue = insurance_data['sex'])
sns.lmplot(x="bmi", y ="charges", hue = "sex",data = insurance_data)
iris_data = pd.read_csv('../input/datafiles/iris.csv', index_col='Id')

iris_data.head()
plt.figure(figsize = (14, 6))



sns.distplot( a= iris_data['Sepal Width (cm)'], kde=False)
plt.figure(figsize = (14, 6))



sns.kdeplot(data = iris_data['Sepal Width (cm)'], shade='True')
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'],kind = 'kde')
iris_set_filepath = "../input/datafiles/iris_setosa.csv"

iris_ver_filepath = "../input/datafiles/iris_versicolor.csv"

iris_vir_filepath = "../input/datafiles/iris_virginica.csv"



iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")

iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")

iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")
sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)

sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)

sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)



plt.title("Distribution of Petal Lengths, by Species")
sns.pairplot(iris_data, hue="Species")