import matplotlib.pyplot as plt

import numpy as np

import sklearn

import sklearn.datasets

import sklearn.linear_model



m = 400 # number of examples

N = int(m/2) # number of points per class



def plot_decision_boundary(model, X, y):

    # Set min and max values and give it some padding

    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1

    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1

    h = 0.01

    # Generate a grid of points with distance h between them

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid

    Z = model(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

    plt.ylabel('x2')

    plt.xlabel('x1')

    plt.scatter(X[0, :], X[1, :], c=y[0], cmap=plt.cm.Spectral)

    



def sigmoid(x):

    """

    Compute the sigmoid of x

    Arguments:

    x -- A scalar or numpy array of any size.

    Return:

    s -- sigmoid(x)

    """

    s = 1/(1+np.exp(-x))

    return s





def load_extra_datasets():  

    N = 200

    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.7, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)

    return  gaussian_quantiles
gaussian_quantiles= load_extra_datasets()

X, Y = gaussian_quantiles

X, Y = X.T, Y.reshape(1, Y.shape[0])

# Visualize the data

plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral);
shape_X = X.shape

shape_Y = Y.shape





print ('The shape of X is: ' + str(shape_X))

print ('The shape of Y is: ' + str(shape_Y))

print ('No of Examples %d!' % (N*2))
# Creating a separable boundary using neural networks



#layer_sizes



def layer_sizes(X, Y):

    """

    Arguments:

    X -- input dataset of shape (input size, number of examples)

    Y -- labels of shape (output size, number of examples)

    

    Returns:

    n_x -- the size of the input layer

    n_h -- the size of the hidden layer

    n_y -- the size of the output layer

    """

    n_x = X.shape[0] # size of input layer`

    n_h = 4

    n_y =Y.shape[0] # size of output layer

    return (n_x, n_h, n_y)
# initialize_parameters



def initialize_parameters(n_x, n_h, n_y):

    """

    Argument:

    n_x -- size of the input layer

    n_h -- size of the hidden layer

    n_y -- size of the output layer

    

    Returns:

    params -- python dictionary containing your parameters:

                    W1 -- weight matrix of shape (n_h, n_x)

                    b1 -- bias vector of shape (n_h, 1)

                    W2 -- weight matrix of shape (n_y, n_h)

                    b2 -- bias vector of shape (n_y, 1)

    """

        

    W1 = np.random.randn(n_h,n_x) * 0.01

    b1 = np.zeros(shape=(n_h, 1))

    W2 = np.random.randn(n_y,n_h) * 0.01

    b2 = np.zeros(shape=(n_y, 1))

    

    

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    

    return parameters
#forward_propagation



def forward_propagation(X, parameters):

    """

    Argument:

    X -- input data of size (n_x, m)

    parameters -- python dictionary containing your parameters (output of initialization function)

    

    Returns:

    A2 -- The sigmoid output of the second activation

    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"

    """

    # Retrieve each parameter from the dictionary "parameters"

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    

    # Implement Forward Propagation to calculate A2 (probabilities)

    Z1 = np.dot(W1,X) + b1

    A1 = np.tanh(Z1)

    Z2A = np.dot(W2,A1)

    Z2 = np.dot(W2,A1) + b2

    A2 = sigmoid(Z2)

    

    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

    

    return A2, cache
# Gcompute_cost



def compute_cost(A2, Y, parameters):

    """

    

    Arguments:

    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)

    Y -- "true" labels vector of shape (1, number of examples)

    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    

    Returns:

    cost -- cross-entropy cost given equation (13)

    """

    

    m = Y.shape[1] # number of example



    # Compute the cross-entropy cost

    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))

    cost = - np.sum(logprobs) / m    

    

    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 

                                # E.g., turns [[17]] into 17 

    

    return cost
# backward_propagation



def backward_propagation(parameters, cache, X, Y):

    """

    Implement the backward propagation using the instructions above.

    

    Arguments:

    parameters -- python dictionary containing our parameters 

    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".

    X -- input data of shape (2, number of examples)

    Y -- "true" labels vector of shape (1, number of examples)

    

    Returns:

    grads -- python dictionary containing your gradients with respect to different parameters

    """

    m = X.shape[1]

    

    # First, retrieve W1 and W2 from the dictionary "parameters".

    W1 = parameters['W1']

    W2 = parameters['W2']

        

    # Retrieve also A1 and A2 from dictionary "cache".

    A1 = cache['A1']

    A2 = cache['A2']

    

    # Backward propagation: calculate dW1, db1, dW2, db2. 



    dZ2 = A2 - Y

    dW2 = (1 / m) * np.dot(dZ2, A1.T)

    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))

    dW1 = (1 / m) * np.dot(dZ1, X.T)

    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    

    grads = {"dW1": dW1,

             "db1": db1,

             "dW2": dW2,

             "db2": db2}

    

    return grads
# update_parameters





def update_parameters(parameters, grads, learning_rate = 1.0):

    """

    Updates parameters using the gradient descent update rule given above

    

    Arguments:

    parameters -- python dictionary containing your parameters 

    grads -- python dictionary containing your gradients 

    

    Returns:

    parameters -- python dictionary containing your updated parameters 

    """

    # Retrieve each parameter from the dictionary "parameters"



    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']



    

    # Retrieve each gradient from the dictionary "grads"

    dW1 = grads['dW1']

    db1 = grads['db1']

    dW2 = grads['dW2']

    db2 = grads['db2']



    

    # Update rule for each parameter



    W1 = W1 - learning_rate * dW1

    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2

    b2 = b2 - learning_rate * db2

    

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    

    return parameters
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):

    

    n_h=n_h-1

    

    """

    Arguments:

    X -- dataset of shape (2, number of examples)

    Y -- labels of shape (1, number of examples)

    n_h -- size of the hidden layer

    num_iterations -- Number of iterations in gradient descent loop

    print_cost -- if True, print the cost every 1000 iterations

    

    Returns:

    parameters -- parameters learnt by the model. They can then be used to predict.

    """

    

    np.random.seed(3)

    n_x = layer_sizes(X, Y)[0]

    n_y = layer_sizes(X, Y)[2]

    

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".

    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    

    # Loop (gradient descent)



    for i in range(0, num_iterations):

         

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".

        A2, cache = forward_propagation(X, parameters)

        

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".

        cost = compute_cost(A2, Y, parameters)

 

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".

        grads = backward_propagation(parameters, cache, X, Y)

 

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".

        parameters = update_parameters(parameters, grads)

        

        # Print the cost every 1000 iterations

        if print_cost and i % 1000 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))



    return parameters
def predict(parameters, X):

    """

    Using the learned parameters, predicts a class for each example in X

    

    Arguments:

    parameters -- python dictionary containing your parameters 

    X -- input data of size (n_x, m)

    

    Returns

    predictions -- vector of predictions of our model (red: 0 / blue: 1)

    """

    

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.

    ### START CODE HERE ### (≈ 2 lines of code)

    A2, cache = forward_propagation(X,parameters)

    predictions = A2 > 0.5

    ### END CODE HERE ###

    

    return predictions
#Now we can run the Model on the entire data with 1 hidden layer (4 neuron units) and 3000 epochs.





# Build a model with a n_h-dimensional hidden layer

parameters = nn_model(X, Y, n_h = 4, num_iterations = 5000, print_cost=True)



# Plot the decision boundary

plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)

plt.title("Decision Boundary for hidden layer size " + str(4));
# Print accuracy

predictions = predict(parameters, X)

print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
#Running the model with diffrent number of neurons in the hiddern layer



plt.figure(figsize=(16, 32))

hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]

for i, n_h in enumerate(hidden_layer_sizes):

    plt.subplot(5, 2, i+1)

    plt.title('Hidden Layer of size %d' % n_h)

    parameters = nn_model(X, Y, n_h, num_iterations = 3000)

    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)

    predictions = predict(parameters, X)

    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)

    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
#Interpretation:



#The larger models (with more hidden units) are able to fit the training set better, until eventually the largest models overfit the data.

#The best hidden layer size seems to be around n_h = 3 without over-fitting. Indeed, a value around here seems to fits the data well without also incurring noticable overfitting.

#We can see that at n_h = 50 the model seems overfitting.