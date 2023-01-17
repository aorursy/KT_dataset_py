import numpy as np

import pandas as pd
def numbify_data(df):

    # Convert sex to numeric

    df['Sex'].replace('female', 0, inplace=True)

    df['Sex'].replace('male', 1, inplace=True)



    # Convert embarked to numeric

    df['Embarked'].replace('Q', 0, inplace=True)

    df['Embarked'].replace('S', 1, inplace=True)

    df['Embarked'].replace('C', 2, inplace=True)



    # Replace nan values by mean

    df.fillna(df.mean(), inplace=True)

    

    

def load_train_set():

    df = pd.read_csv('../input/train.csv', delimiter = ',')

    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    numbify_data(df)

    train_set_x = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values.T

    train_set_y = df[['Survived']].values.T

    return train_set_x, train_set_y



def load_test_set():

    df = pd.read_csv('../input/test.csv', delimiter = ',')

    df = df[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    numbify_data(df)

    test_set_x = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values.T

    test_set_id = df[['PassengerId']].values.T

    return test_set_x, test_set_id



def sigmoid(z):

    s = 1/(1 + np.exp(-z))

    return s



def relu(x):

    return np.abs(x) * (x > 0)
train_set_x, train_set_y = load_train_set()

test_set_x, test_set_id = load_test_set()
m_train = train_set_x.shape[1]

m_test = test_set_x.shape[1]

print ("Number of training examples: m_train = " + str(m_train))

print ("Number of testing examples: m_test = " + str(m_test))


def initialize_with_zeros(dim_features):

    w = np.zeros((dim_features, 1))

    b = 0

    return w, b
dim_features = train_set_x.shape[0]

w, b = initialize_with_zeros(dim_features)

print ("w = " + str(w))

print ("b = " + str(b))
def propagate(w, b, X, Y):

    m = X.shape[1]

    #forward propagation

    A = sigmoid(np.dot(w.T, X) + b)

    cost = (-1/m) * np.sum(Y*np.log(A) + (1 - Y)*np.log(1-A))

    

    #backward propagation

    dw = (1/m) * np.dot(X, (A-Y).T)

    db = (1/m) * np.sum(A-Y)

    

    grads = {'dw': dw,

            'db': db}

    return grads, cost

    
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        w = w - learning_rate*grads['dw']

        b = b - learning_rate*grads['db']

        if print_cost and i % 1000 == 0:

            print("cost after iteration", i,":", cost)

        costs.append(cost)

    params = {'w': w,

             'b': b}



    grads = {'dw': grads['dw'],

             'db': grads['db']}

    

    return params, grads, costs
X = train_set_x

Y = train_set_y

w, b = initialize_with_zeros(X.shape[0])

params, grads, costs = optimize(w, b, X, Y, num_iterations= 10000, learning_rate = 0.004, print_cost = True)



print ("w = " + str(params["w"]))

print ("b = " + str(params["b"]))
def predict(w, b, X):

    m = X.shape[1]

    Y_prediction = np.zeros((1,m))

    w = w.reshape(X.shape[0], 1)

    

    A = sigmoid(np.dot(w.T,X) + b)

    

    for i in range(A.shape[1]):

        Y_prediction[0, i] = 1 if A[0, i] >= .5 else 0 

        pass

    return Y_prediction
X_train = train_set_x

Y_train = train_set_y



np.nan_to_num(X_train, copy=False)

np.nan_to_num(Y_train, copy=False)



X_test = test_set_x

np.nan_to_num(X_test, copy=False)



Y_prediction_test = predict(w,b,X_test)

Y_prediction_train = predict(w,b,X_train)



print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

np.savetxt('submission.csv', np.hstack([test_set_id.T, Y_prediction_test.T]), delimiter=',')
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

    n_x = X.shape[0] # size of input layer

    n_h = 5

    n_y = Y.shape[0] # size of output layer

    return (n_x, n_h, n_y)
(n_x, n_h, n_y) = layer_sizes(X_train, Y_train)

print("The size of the input layer is: n_x = " + str(n_x))

print("The size of the hidden layer is: n_h = " + str(n_h))

print("The size of the output layer is: n_y = " + str(n_y))
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

    W1 = (np.random.randn(n_h,n_x)*0.01).reshape((n_h, n_x))

    b1 = np.zeros((n_h,1))

    W2 = (np.random.rand(n_y,n_h)*0.01).reshape((n_y, n_h))

    b2 = np.zeros((n_y, 1))

    

    assert (W1.shape == (n_h, n_x))

    assert (b1.shape == (n_h, 1))

    assert (W2.shape == (n_y, n_h))

    assert (b2.shape == (n_y, 1))

    

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    

    return parameters
parameters = initialize_parameters(n_x, n_h, n_y)

print("W1 = " + str(parameters["W1"]))

print("b1 = " + str(parameters["b1"]))

print("W2 = " + str(parameters["W2"]))

print("b2 = " + str(parameters["b2"]))
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

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

    

    # Implement Forward Propagation to calculate A2 (probabilities)

    Z1 = np.dot(W1,X) + b1

    A1 = np.tanh(Z1)

    Z2 = np.dot(W2,A1) + b2

    A2 = sigmoid(Z2)

    

    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

    

    return A2, cache
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

    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1 - A2),1 - Y)

    cost = -1/m * np.sum(logprobs)

    

    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 

                                # E.g., turns [[17]] into 17 

    

    return cost
def backward_propagation(parameters, cache, X, Y):

    """

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

    W1 = parameters["W1"]

    W2 = parameters["W2"]

        

    # Retrieve also A1 and A2 from dictionary "cache".

    A1 = cache["A1"]

    A2 = cache["A2"]

    

    # Backward propagation: calculate dW1, db1, dW2, db2. 

    dZ2 = A2 - Y

    dW2 = 1/m * np.dot(dZ2,A1.T)

    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)

    dZ1 = np.dot(W2.T,dZ2)*(1 - np.power(A1,2))

    dW1 = 1/m * np.dot(dZ1,X.T)

    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)

    

    grads = {"dW1": dW1,

             "db1": db1,

             "dW2": dW2,

             "db2": db2}

    

    return grads
def update_parameters(parameters, grads, learning_rate = 0.005):

    """

    Arguments:

    parameters -- python dictionary containing your parameters 

    grads -- python dictionary containing your gradients 

    

    Returns:

    parameters -- python dictionary containing your updated parameters 

    """

    # Retrieve each parameter from the dictionary "parameters"

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

    

    # Retrieve each gradient from the dictionary "grads"

    dW1 = grads["dW1"]

    db1 = grads["db1"]

    dW2 = grads["dW2"]

    db2 = grads["db2"]

    

    # Update rule for each parameter

    W1 = W1 - learning_rate*dW1

    b1 = b1 - learning_rate*db1

    W2 = W2 - learning_rate*dW2

    b2 = b2 - learning_rate*db2

    

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    

    return parameters
def nn_model(X, Y, n_h, num_iterations = 10000, learning_rate = 0.004, print_cost=False):

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

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

    

    # Loop (gradient descent)



    for i in range(0, num_iterations):

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".

        A2, cache = forward_propagation(X, parameters)

        

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".

        cost = compute_cost(A2,Y,parameters)

 

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".

        grads = backward_propagation(parameters, cache, X, Y)

 

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".

        parameters = update_parameters(parameters, grads, learning_rate)

        

        # Print the cost every 10000 iterations

        if print_cost and i % 10000 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))



    return parameters
parameters = nn_model(X_train, Y_train, 5, num_iterations=100000, learning_rate=0.004, print_cost=True)
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

    A2, cache = forward_propagation(X, parameters)

    predictions = A2 > .5

    predictions = predictions.astype(int)

    

    return predictions
predictions_train = predict(parameters, X_train)

print ('Accuracy on train set: %d' % float((np.dot(Y,predictions_train.T) + np.dot(1-Y,1-predictions_train.T))/float(Y.size)*100) + '%')

predictions_test = predict(parameters, X_test)

df = pd.DataFrame({'PassengerId':np.squeeze(test_set_id), 'Survived':np.squeeze(predictions_test)})

df.to_csv('submission.csv',index=False)