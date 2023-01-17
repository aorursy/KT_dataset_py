import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
iris = load_iris()
X,y = iris['data'],iris['target']
def to_one_hot(y):

    n_classes = y.max() + 1

    m = len(y)

    Y_one_hot = np.zeros((m, n_classes))

    Y_one_hot[np.arange(m), y] = 1

    return Y_one_hot
def softmax(z):

    exps = np.exp(z)

    exps_sum = np.sum(exps,axis=1,keepdims=True)

    return exps / exps_sum
def cross_entropy(y_proba,y_true):

    return -(1/m) * np.sum(y_true * np.log(y_proba+epsilon))
X_b = np.c_[np.ones([len(X),1]),X]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
n_inputs = X_train.shape[1]

n_outputs = len(np.unique(y_train))
thetas = np.random.randn(n_inputs,n_outputs)

m = len(X_train)
eta = 0.1

epsilon = 1e-7
y_train_one_hot = to_one_hot(y_train)
def batch_gradient_descent(X_train,y_train,thetas,alpha=eta,n_iters=5000):

    for i in range(n_iters):

        logits = X_train.dot(thetas)

        y_proba = softmax(logits)

        loss = cross_entropy(y_proba,y_train_one_hot)

        error = y_proba - y_train_one_hot

        thetas = thetas - (alpha/m) * X_train.T.dot(error)

    return thetas
n_th = batch_gradient_descent(X_train,y_train,thetas)
test_logits = X_test.dot(n_th)

test_proba = softmax(test_logits)

test_predictions = np.argmax(test_proba,axis=1)
accuracy_score = np.mean(test_predictions == y_test)

round(accuracy_score,2)