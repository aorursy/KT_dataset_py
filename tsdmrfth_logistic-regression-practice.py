import pandas as pd

import numpy as np

from sklearn.datasets import make_classification

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=1,

                             n_clusters_per_class=1, random_state=14)



y = y[:,np.newaxis]

m = len(y)



sns.set_style('white')

sns.scatterplot(X[:,0],X[:,1],hue=y.reshape(-1));



X = np.hstack((np.ones((m,1)),X)) # Add extra feature for theta0 (theta0 is always equals to 1)

n = np.size(X,1)



m = len(y)

theta = np.zeros((n,1))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
def sigmoid(z):

    return 1 / (1 + np.exp(-z))
def get_cost(X, y, theta):

    h = sigmoid(X @ theta)

    epsilon = 1e-5

    cost = (1 / m) * (((-y).T @ np.log(h + epsilon))-((1 - y).T @ np.log(1- h + epsilon)))

    return cost
def get_gradient(X, y, theta):

    return 1/m * (X.T @ (sigmoid(X @ theta) - y))
def gradient_descent(X, y, theta, learning_rate, iterations):

    m = len(y)

    cost_history = np.zeros((iterations,1))



    for i in range(iterations):

        theta = theta - learning_rate * get_gradient(X, y, theta)

        cost_history[i] = get_cost(X, y, theta)



    return (cost_history, theta)
iterations = 1500

learning_rate = 0.03



initial_cost = compute_cost(X, y, params)



print("Initial Cost is: {} \n".format(initial_cost))



(cost_history, optional_thetas) = gradient_descent(X_train, y_train, params, learning_rate, iterations)



print("Optimal Parameters are: \n", optional_thetas, "\n")



fig = plt.figure()

fig.set_size_inches(8, 5)

sns.set_style('white')

plt.plot(range(len(cost_history)), cost_history, 'r')

plt.title("Convergence Graph of Cost Function")

plt.xlabel("Number of Iterations")

plt.ylabel("Cost")

plt.show()
def get_predictions(X, theta):

    predictions = sigmoid(X @ theta)

    return [1 if prediction >= 0.5 else 0 for prediction in predictions]
actual_yes = 'Actual Yes'

actual_no = 'Actual No'

predicted_yes = 'Predicted Yes'

predicted_no = 'Predicted No'



def get_confusion_matrix():

    predictions = get_predictions(X_test, optional_thetas)

    actual_values = y_test

    confusion_matrix = pd.DataFrame(columns=[actual_yes, actual_no], index=[predicted_yes, predicted_no])

    confusion_matrix = confusion_matrix.fillna(0)

    

    for i, prediction in enumerate(predictions):

        if actual_values[i] == 1:

            if prediction == 1:

                confusion_matrix[actual_yes][predicted_yes] += 1

            else:

                confusion_matrix[actual_yes][predicted_no] += 1

        else:

            if prediction == 1:

                confusion_matrix[actual_no][predicted_yes] += 1

            else:

                confusion_matrix[actual_no][predicted_no] += 1

        

    return confusion_matrix
confusion_matrix = get_confusion_matrix()
confusion_matrix
N = 2

predicted_yes_values = (confusion_matrix[actual_yes][predicted_yes], confusion_matrix[actual_yes][predicted_no])

predicted_no_values = (confusion_matrix[actual_no][predicted_yes], confusion_matrix[actual_no][predicted_no])



ind = np.arange(N)

width = 0.20      

fig, ax = plt.subplots()

rect1 = plt.bar(ind, predicted_yes_values, width, label=actual_yes)

rect2 = plt.bar(ind + width, predicted_no_values, width, label=actual_no)



plt.ylabel('Scores')

plt.title('Actual classes and predicted classes')



plt.xticks(ind + width / 2, (predicted_yes, predicted_no))

plt.legend(loc='best')



fig.set_size_inches(8, 8)



def auto_label(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width() / 2., 1.005 * height,

                '%d' % int(height),

                ha='center', va='bottom')



auto_label(rect1)

auto_label(rect2)



plt.show()