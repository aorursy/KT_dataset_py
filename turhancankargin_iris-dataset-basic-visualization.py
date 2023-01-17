import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Learning Matplotlib Library (# Visualization Library)



# Learning Line plot, Scatter plot, Histogram, and Subplots.

df = pd.read_csv("/kaggle/input/iris/Iris.csv")



# Lets observe variables

print(df.columns)

print(df.Species.unique())
# More details

print(df.info())

print(df.describe())
iris_setosa = df[df.Species == "Iris-setosa"]

iris_versicolor = df[df.Species == "Iris-versicolor"]

iris_virginica = df[df.Species == "Iris-virginica"]

print(iris_setosa.describe())

print(iris_versicolor.describe())

print(iris_virginica.describe())
# Line Plot

import matplotlib.pyplot as plt # Import Library

df1 = df.drop(["Id"],axis=1) # Drop the Id column

plt.plot(iris_setosa.Id,iris_setosa.PetalLengthCm,color="red",label= "Iris Setosa")

plt.plot(iris_versicolor.Id,iris_versicolor.PetalLengthCm,color="green",label= "Iris Versicolor")

plt.plot(iris_virginica.Id,iris_virginica.PetalLengthCm,color="blue",label= "Iris Virginica")

plt.legend()

plt.title("ID of Species versus Petal Length")

plt.xlabel("Id")

plt.ylabel("PetalLengthCm")

plt.show()
# Showing grid

df1.plot(grid=True,alpha= 0.9)

plt.show()
# Scatter plot

plt.scatter(iris_setosa.PetalLengthCm,iris_setosa.PetalWidthCm,color="red",label="iris_setosa")

plt.scatter(iris_versicolor.PetalLengthCm,iris_versicolor.PetalWidthCm,color="orange",label="iris_versicolor")

plt.scatter(iris_virginica.PetalLengthCm,iris_virginica.PetalWidthCm,color="darkblue",label="iris_virginica")

plt.legend()

plt.xlabel("Petal Length (cm)")

plt.ylabel("Petal Width (cm)")

plt.title("Scatter Plot")

plt.show()
# Histogram Plot

plt.hist(iris_setosa.PetalLengthCm,bins= 20)

plt.xlabel("PetalLength (cm)")

plt.ylabel("Frequency")

plt.title("Histogram")

plt.show()
# Subplots

df1.plot(grid=True,alpha= 0.9,subplots = True)

plt.show()
# Another Method

plt.subplot(2,1,1)

plt.plot(iris_setosa.Id,iris_setosa.PetalLengthCm,color="red",label= "Iris Setosa")

plt.ylabel("Setosa - P.L. (cm)")

plt.subplot(2,1,2)

plt.plot(iris_versicolor.Id,iris_versicolor.PetalLengthCm,color="green",label= "Iris Versicolor")

plt.ylabel("Versicolor - P.L. (cm)")

plt.show()