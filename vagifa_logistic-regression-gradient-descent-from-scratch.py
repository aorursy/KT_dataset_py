import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
data = load_breast_cancer()
df = pd.DataFrame(data['data'],columns=data['feature_names'])
df
X,y = data['data'],data['target']
X_scaled = (X-X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X = X_scaled
X_b = np.c_[np.ones((len(X),1)),X]
X = X_b
scaler = MinMaxScaler()

X_scaled_test = scaler.fit_transform(X)
X
X_scaled_test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
thetas = np.zeros(df.shape[1]+1)
m = len(y)
def cost_function(X,y,thetas):
    predictions = sigmoid(np.dot(X,thetas.T))
    J = -(1/m) * np.sum(y*np.log(predictions) + (1-y)*np.log(1-predictions))
    return J
cost_function(X,y,thetas)
def batch_gradient_descent(X,y,thetas,alpha=0.5,n_iters=2500):
    c_hist = [0] * n_iters
    
    for i in range(n_iters):
        prediction = sigmoid(np.dot(X,thetas.T))
        thetas = thetas - (alpha/m) * np.dot(prediction-y,X)
        c_hist[i] = cost_function(X,y,thetas)
        
    return thetas,c_hist
batch_gd_thetas,batch_gd_cost = batch_gradient_descent(X_train,y_train,thetas)
import matplotlib.pyplot as plt

plt.plot(range(2500),batch_gd_cost)
plt.xlabel('No. of Iterations')
plt.ylabel('J (θ)')
plt.title('Batch Gradient Descent')
plt.show()
threshold = 0.5

final_predictions = pd.Series(sigmoid(np.dot(X_test,batch_gd_thetas.T)) >= threshold).astype('int')

print(classification_report(y_test,final_predictions))
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train,y_train)
print(classification_report(y_test,lg.predict(X_test)))
