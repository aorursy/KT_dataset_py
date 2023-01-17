import numpy as np

import h5py

import matplotlib.pyplot as plt

from testCases_v4a import *

from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward



%matplotlib inline

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



%load_ext autoreload

%autoreload 2



np.random.seed(1)
# GRADED FUNCTION: initialize_parameters



def initialize_parameters(n_x, n_h, n_y):

    """

    Argument:

    n_x -- size of the input layer

    n_h -- size of the hidden layer

    n_y -- size of the output layer

    

    Returns:

    parameters -- python dictionary containing your parameters:

                    W1 -- weight matrix of shape (n_h, n_x)

                    b1 -- bias vector of shape (n_h, 1)

                    W2 -- weight matrix of shape (n_y, n_h)

                    b2 -- bias vector of shape (n_y, 1)

    """

    

    np.random.seed(1)

    

    ### START CODE HERE ### (≈ 4 lines of code)

    W1 = None

    b1 = None

    W2 = None

    b2 = None

    ### END CODE HERE ###

    

    assert(W1.shape == (n_h, n_x))

    assert(b1.shape == (n_h, 1))

    assert(W2.shape == (n_y, n_h))

    assert(b2.shape == (n_y, 1))

    

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    

    return parameters    
parameters = initialize_parameters(3,2,1)

print("W1 = " + str(parameters["W1"]))

print("b1 = " + str(parameters["b1"]))

print("W2 = " + str(parameters["W2"]))

print("b2 = " + str(parameters["b2"]))
# GRADED FUNCTION: initialize_parameters_deep



def initialize_parameters_deep(layer_dims):

    """

    Arguments:

    layer_dims -- python array (list) containing the dimensions of each layer in our network

    

    Returns:

    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":

                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])

                    bl -- bias vector of shape (layer_dims[l], 1)

    """

    

    np.random.seed(3)

    parameters = {}

    L = len(layer_dims)            # number of layers in the network



    for l in range(1, L):

        ### START CODE HERE ### (≈ 2 lines of code)

        parameters['W' + str(l)] = None

        parameters['b' + str(l)] = None

        ### END CODE HERE ###

        

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))

        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))



        

    return parameters
parameters = initialize_parameters_deep([5,4,3])

print("W1 = " + str(parameters["W1"]))

print("b1 = " + str(parameters["b1"]))

print("W2 = " + str(parameters["W2"]))

print("b2 = " + str(parameters["b2"]))
# GRADED FUNCTION: linear_forward



def linear_forward(A, W, b):

    """

    Implement the linear part of a layer's forward propagation.



    Arguments:

    A -- activations from previous layer (or input data): (size of previous layer, number of examples)

    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)

    b -- bias vector, numpy array of shape (size of the current layer, 1)



    Returns:

    Z -- the input of the activation function, also called pre-activation parameter 

    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently

    """

    

    ### START CODE HERE ### (≈ 1 line of code)

    Z = None

    ### END CODE HERE ###

    

    assert(Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)

    

    return Z, cache
A, W, b = linear_forward_test_case()



Z, linear_cache = linear_forward(A, W, b)

print("Z = " + str(Z))
# GRADED FUNCTION: linear_activation_forward



def linear_activation_forward(A_prev, W, b, activation):

    """

    Implement the forward propagation for the LINEAR->ACTIVATION layer



    Arguments:

    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)

    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)

    b -- bias vector, numpy array of shape (size of the current layer, 1)

    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"



    Returns:

    A -- the output of the activation function, also called the post-activation value 

    cache -- a python tuple containing "linear_cache" and "activation_cache";

             stored for computing the backward pass efficiently

    """

    

    if activation == "sigmoid":

        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".

        ### START CODE HERE ### (≈ 2 lines of code)

        Z, linear_cache = None

        A, activation_cache = None

        ### END CODE HERE ###

    

    elif activation == "relu":

        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".

        ### START CODE HERE ### (≈ 2 lines of code)

        Z, linear_cache = None

        A, activation_cache = None

        ### END CODE HERE ###

    

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)



    return A, cache
A_prev, W, b = linear_activation_forward_test_case()



A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")

print("With sigmoid: A = " + str(A))



A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")

print("With ReLU: A = " + str(A))
# GRADED FUNCTION: L_model_forward



def L_model_forward(X, parameters):

    """

    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    

    Arguments:

    X -- data, numpy array of shape (input size, number of examples)

    parameters -- output of initialize_parameters_deep()

    

    Returns:

    AL -- last post-activation value

    caches -- list of caches containing:

                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)

    """



    caches = []

    A = X

    L = len(parameters) // 2                  # number of layers in the neural network

    

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.

    for l in range(1, L):

        A_prev = A 

        ### START CODE HERE ### (≈ 2 lines of code)

        A, cache = None

        None

        ### END CODE HERE ###

    

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.

    ### START CODE HERE ### (≈ 2 lines of code)

    AL, cache = None

    None

    ### END CODE HERE ###

    

    assert(AL.shape == (1,X.shape[1]))

            

    return AL, caches
X, parameters = L_model_forward_test_case_2hidden()

AL, caches = L_model_forward(X, parameters)

print("AL = " + str(AL))

print("Length of caches list = " + str(len(caches)))
# GRADED FUNCTION: compute_cost



def compute_cost(AL, Y):

    """

    Implement the cost function defined by equation (7).



    Arguments:

    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)

    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)



    Returns:

    cost -- cross-entropy cost

    """

    

    m = Y.shape[1]



    # Compute loss from aL and y.

    ### START CODE HERE ### (≈ 1 lines of code)

    cost = None

    ### END CODE HERE ###

    

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    assert(cost.shape == ())

    

    return cost
Y, AL = compute_cost_test_case()



print("cost = " + str(compute_cost(AL, Y)))
# GRADED FUNCTION: linear_backward



def linear_backward(dZ, cache):

    """

    Implement the linear portion of backward propagation for a single layer (layer l)



    Arguments:

    dZ -- Gradient of the cost with respect to the linear output (of current layer l)

    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer



    Returns:

    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev

    dW -- Gradient of the cost with respect to W (current layer l), same shape as W

    db -- Gradient of the cost with respect to b (current layer l), same shape as b

    """

    A_prev, W, b = cache

    m = A_prev.shape[1]



    ### START CODE HERE ### (≈ 3 lines of code)

    dW = None

    db = None

    dA_prev = None

    ### END CODE HERE ###

    

    assert (dA_prev.shape == A_prev.shape)

    assert (dW.shape == W.shape)

    assert (db.shape == b.shape)

    

    return dA_prev, dW, db
# Set up some test inputs

dZ, linear_cache = linear_backward_test_case()



dA_prev, dW, db = linear_backward(dZ, linear_cache)

print ("dA_prev = "+ str(dA_prev))

print ("dW = " + str(dW))

print ("db = " + str(db))
# GRADED FUNCTION: linear_activation_backward



def linear_activation_backward(dA, cache, activation):

    """

    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    

    Arguments:

    dA -- post-activation gradient for current layer l 

    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently

    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    

    Returns:

    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev

    dW -- Gradient of the cost with respect to W (current layer l), same shape as W

    db -- Gradient of the cost with respect to b (current layer l), same shape as b

    """

    linear_cache, activation_cache = cache

    

    if activation == "relu":

        ### START CODE HERE ### (≈ 2 lines of code)

        dZ = None

        dA_prev, dW, db = None

        ### END CODE HERE ###

        

    elif activation == "sigmoid":

        ### START CODE HERE ### (≈ 2 lines of code)

        dZ = None

        dA_prev, dW, db = None

        ### END CODE HERE ###

    

    return dA_prev, dW, db
dAL, linear_activation_cache = linear_activation_backward_test_case()



dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")

print ("sigmoid:")

print ("dA_prev = "+ str(dA_prev))

print ("dW = " + str(dW))

print ("db = " + str(db) + "\n")



dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")

print ("relu:")

print ("dA_prev = "+ str(dA_prev))

print ("dW = " + str(dW))

print ("db = " + str(db))
# GRADED FUNCTION: L_model_backward



def L_model_backward(AL, Y, caches):

    """

    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    

    Arguments:

    AL -- probability vector, output of the forward propagation (L_model_forward())

    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)

    caches -- list of caches containing:

                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)

                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    

    Returns:

    grads -- A dictionary with the gradients

             grads["dA" + str(l)] = ... 

             grads["dW" + str(l)] = ...

             grads["db" + str(l)] = ... 

    """

    grads = {}

    L = len(caches) # the number of layers

    m = AL.shape[1]

    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    

    # Initializing the backpropagation

    ### START CODE HERE ### (1 line of code)

    dAL = None

    ### END CODE HERE ###

    

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]

    ### START CODE HERE ### (approx. 2 lines)

    current_cache = None

    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = None

    ### END CODE HERE ###

    

    # Loop from l=L-2 to l=0

    for l in reversed(range(L-1)):

        # lth layer: (RELU -> LINEAR) gradients.

        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 

        ### START CODE HERE ### (approx. 5 lines)

        current_cache = None

        dA_prev_temp, dW_temp, db_temp = None

        grads["dA" + str(l)] = None

        grads["dW" + str(l + 1)] = None

        grads["db" + str(l + 1)] = None

        ### END CODE HERE ###



    return grads
AL, Y_assess, caches = L_model_backward_test_case()

grads = L_model_backward(AL, Y_assess, caches)

print_grads(grads)
# GRADED FUNCTION: update_parameters



def update_parameters(parameters, grads, learning_rate):

    """

    Update parameters using gradient descent

    

    Arguments:

    parameters -- python dictionary containing your parameters 

    grads -- python dictionary containing your gradients, output of L_model_backward

    

    Returns:

    parameters -- python dictionary containing your updated parameters 

                  parameters["W" + str(l)] = ... 

                  parameters["b" + str(l)] = ...

    """

    

    L = len(parameters) // 2 # number of layers in the neural network



    # Update rule for each parameter. Use a for loop.

    ### START CODE HERE ### (≈ 3 lines of code)

    for l in range(L):

        parameters["W" + str(l+1)] = None

        parameters["b" + str(l+1)] = None

    ### END CODE HERE ###

    return parameters
parameters, grads = update_parameters_test_case()

parameters = update_parameters(parameters, grads, 0.1)



print ("W1 = "+ str(parameters["W1"]))

print ("b1 = "+ str(parameters["b1"]))

print ("W2 = "+ str(parameters["W2"]))

print ("b2 = "+ str(parameters["b2"]))