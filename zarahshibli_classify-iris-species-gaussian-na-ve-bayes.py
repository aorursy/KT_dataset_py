import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import seaborn as sns



def p_x_given_y(x, mean_y, variance_y):



    # Input the arguments into a probability density function

    p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))

    

    # return p

    return p
data = pd.read_csv("../input/Iris.csv")

data = data.drop(columns="Id")

train, test = train_test_split(data, train_size =0.7)

test.head()
sns.pairplot(data=data,hue='Species',x_vars=['SepalLengthCm'],y_vars=['SepalWidthCm'],height=5)


# Total number of example in test set 

total = train["Species"].count()



# Number of example has setosa class 

n_setosa = train["Species"].value_counts()["Iris-setosa"]



# Number of example has versicolor class 

n_versicolor = train["Species"].value_counts()["Iris-versicolor"]



# Number of example has virginica class 

n_virginica = train["Species"].value_counts()["Iris-virginica"]





#P(setosa)

priors_setosa = n_setosa / total



#P(versicolor)

priors_versicolor = n_versicolor / total



#P(virginica)

priors_virginica = n_virginica / total

# Group the data by species and calculate the means of each feature

data_means = train.groupby('Species').mean()



# View the mean values

data_means

# Group the data by species and calculate the variance of each feature

data_variance = data.groupby('Species').var()



# View the variance values

data_variance




# Mean for setosa

setosa_sepal_length_mean = data_means['SepalLengthCm'][data_variance.index == 'Iris-setosa'].values[0]

setosa_sepal_width_mean = data_means['SepalWidthCm'][data_variance.index == 'Iris-setosa'].values[0]

setosa_petal_length_mean = data_means['PetalLengthCm'][data_variance.index == 'Iris-setosa'].values[0]

setosa_petal_width_mean = data_means['PetalWidthCm'][data_variance.index == 'Iris-setosa'].values[0]



# Variance for setosa

setosa_sepal_length_variance = data_variance['SepalLengthCm'][data_variance.index == 'Iris-setosa'].values[0]

setosa_sepal_width_variance = data_variance['SepalWidthCm'][data_variance.index == 'Iris-setosa'].values[0]

setosa_petal_length_variance = data_variance['PetalLengthCm'][data_variance.index == 'Iris-setosa'].values[0]

setosa_petal_width_variance = data_variance['PetalWidthCm'][data_variance.index == 'Iris-setosa'].values[0]





# Mean for virginica

virginica_sepal_length_mean = data_means['SepalLengthCm'][data_variance.index == 'Iris-virginica'].values[0]

virginica_sepal_width_mean = data_means['SepalWidthCm'][data_variance.index == 'Iris-virginica'].values[0]

virginica_petal_length_mean = data_means['PetalLengthCm'][data_variance.index == 'Iris-virginica'].values[0]

virginica_petal_width_mean = data_means['PetalWidthCm'][data_variance.index == 'Iris-virginica'].values[0]





# Variance for virginica

virginica_sepal_length_variance = data_variance['SepalLengthCm'][data_variance.index == 'Iris-virginica'].values[0]

virginica_sepal_width_variance = data_variance['SepalWidthCm'][data_variance.index == 'Iris-virginica'].values[0]

virginica_petal_length_variance = data_variance['PetalLengthCm'][data_variance.index == 'Iris-virginica'].values[0]

virginica_petal_width_variance = data_variance['PetalWidthCm'][data_variance.index == 'Iris-virginica'].values[0]



# Mean for versicolor

versicolor_sepal_length_mean = data_means['SepalLengthCm'][data_variance.index == 'Iris-versicolor'].values[0]

versicolor_sepal_width_mean = data_means['SepalWidthCm'][data_variance.index == 'Iris-versicolor'].values[0]

versicolor_petal_length_mean = data_means['PetalLengthCm'][data_variance.index == 'Iris-versicolor'].values[0]

versicolor_petal_width_mean = data_means['PetalWidthCm'][data_variance.index == 'Iris-versicolor'].values[0]



# Variance for versicolor

versicolor_sepal_length_variance = data_variance['SepalLengthCm'][data_variance.index == 'Iris-versicolor'].values[0]

versicolor_sepal_width_variance = data_variance['SepalWidthCm'][data_variance.index == 'Iris-versicolor'].values[0]

versicolor_petal_length_variance = data_variance['PetalLengthCm'][data_variance.index == 'Iris-versicolor'].values[0]

versicolor_petal_width_variance = data_variance['PetalWidthCm'][data_variance.index == 'Iris-versicolor'].values[0]







#P(setosa| x)

P_of_setosa = priors_setosa * p_x_given_y(test['SepalLengthCm'], setosa_sepal_length_mean, setosa_sepal_length_variance) \

* p_x_given_y(test['SepalWidthCm'], setosa_sepal_width_mean, setosa_sepal_width_variance) \

* p_x_given_y(test['PetalLengthCm'], setosa_petal_length_mean, setosa_petal_length_variance) \

* p_x_given_y(test['PetalWidthCm'], setosa_petal_width_mean, setosa_petal_width_variance)



# P(virginica| x)

P_of_virginica = priors_virginica * p_x_given_y(test['SepalLengthCm'], virginica_sepal_length_mean, virginica_sepal_length_variance) \

* p_x_given_y(test['SepalWidthCm'], virginica_sepal_width_mean, virginica_sepal_width_variance) \

* p_x_given_y(test['PetalLengthCm'], virginica_petal_length_mean, virginica_petal_length_variance) \

* p_x_given_y(test['PetalWidthCm'], virginica_petal_width_mean, virginica_petal_width_variance)





# P(versicolor| x)

P_of_versicolor = priors_versicolor * p_x_given_y(test['SepalLengthCm'], versicolor_sepal_length_mean, versicolor_sepal_length_variance) \

* p_x_given_y(test['SepalWidthCm'], versicolor_sepal_width_mean, versicolor_sepal_width_variance) \

* p_x_given_y(test['PetalLengthCm'], versicolor_petal_length_mean, versicolor_petal_length_variance) \

* p_x_given_y(test['PetalWidthCm'], versicolor_petal_width_mean, versicolor_petal_width_variance)



predicted = []

for x, y, z in zip(P_of_setosa, P_of_virginica,P_of_versicolor):

    if(x > y and x > z):

        predicted.append("Iris-setosa")

    if(y > x and y > z):

        predicted.append("Iris-virginica")

    if(z > x and z > y):

        predicted.append("Iris-versicolor")

        

correct = 0

actual = test["Species"]



for x, y in zip(actual, predicted):

    if x == y:

        correct += 1

print("The accuracy is {:.2f}".format((correct / float(len(test))*100)),"%") 