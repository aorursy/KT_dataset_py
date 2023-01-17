import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as sklearn_lr

import warnings
warnings.simplefilter('ignore')
seed = 13
data = pd.read_csv("../input/qprereq/assignment2_data.csv", names=['X', 'y'])
print(f"Data Shape : {data.shape}")

data.head()
class LinearRegression():
    
    def __init__(self, lr, n_iter, normalize_f):
        self.lr = lr
        self.n_iter = n_iter
        self.coef_ = None
        self.intercept_ = None
        self.normalize_f = normalize_f
    
    def init_constants(self, X, y):
        self.X = X
        if self.normalize_f:
            self.X = self.normalize(self.X)
        self.X = self.add_bias(self.X)
        
        self.y = y
        self.n_samples = self.y.shape[0]
        
        self.n_features = X.shape[1]
        self.params = np.zeros((self.n_features + 1, 1))
    
    def normalize(self, X):
        mu = np.mean(X, 0)
        std = np.std(X, 0)
        
        X = (X - mu)/std
        
        return X
    
    def add_bias(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        return X
    
    def fit(self, X, y):
        self.init_constants(X, y)
        
        for i in range(self.n_iter):
            self.params = self.params - (self.lr/self.n_samples) * (self.X.T @ (self.X @ self.params - self.y))
        
        self.coef_ = self.params[1: ]
        self.intercept_ = self.params[0]
        
        return self
    
    def predict(self, X):
        X = self.normalize(X)
        X = self.add_bias(X)
        
        y_pred = X @ self.params
        
        return y_pred
    
    def get_params(self):
        return self.params
    
    def plot(self):
        pass
    
    def r2_score(self, X=None, y=None):
        if X is None:
            X = self.X
            y = self.y
        else:
            X = self.normalize(X)
            X = self.add_bias(X)
        
        y_pred = X @ self.params
        score = 1 - (((y - y_pred)**2).sum() / ((y - y.mean())**2).sum())
        
        return score
X = data[['X']].values
y = data[['y']].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
lr = 0.01
n_iter = 1_000
normalize = True

mine = LinearRegression(lr=lr, n_iter=n_iter, normalize_f=normalize)
skl = sklearn_lr(normalize=normalize)

mine.fit(X_train, y_train)
skl.fit(X_train, y_train)
plt.scatter(X_train, y_train, color='blue', label='Train')
plt.scatter(X_train, mine.predict(X_train), color='red', label='Mine')
plt.scatter(X_train, skl.predict(X_train), color='green', label='Skl')

plt.legend()
plt.show()
plt.scatter(X_valid, y_valid, color='blue', label='Valid')
plt.scatter(X_valid, mine.predict(X_valid), color='red', label='Mine')
plt.scatter(X_valid, skl.predict(X_valid), color='green', label='Skl')

plt.legend()
plt.show()
mine_r2_train = mine.r2_score(X_train, y_train)
mine_r2_valid = mine.r2_score(X_valid, y_valid)

skl_r2_train = skl.score(X_train, y_train)
skl_r2_valid = skl.score(X_valid, y_valid)

print(f"R2 \t\tTrain\t\t\tValid \nMine : \t{mine_r2_train}\t{mine_r2_valid}\nSKL  : \t{skl_r2_train}\t{skl_r2_valid}")
def mae(y_true, y_pred):
    return np.mean(y_true - y_pred)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def eval_metric(func, model, X, y):
    return func(y, model.predict(X))
# MAE

mine_mae_train = eval_metric(mae, mine, X_train, y_train)
mine_mae_valid = eval_metric(mae, mine, X_valid, y_valid)

skl_mae_train = eval_metric(mae, skl, X_train, y_train)
skl_mae_valid = eval_metric(mae, skl, X_valid, y_valid)

print(f"MAE \t\tTrain\t\t\tValid \nMine : \t{mine_mae_train}\t{mine_mae_valid}\nSKL  : \t{skl_mae_train}\t{skl_mae_valid}")
# MSE

mine_mse_train = eval_metric(mse, mine, X_train, y_train)
mine_mse_valid = eval_metric(mse, mine, X_valid, y_valid)

skl_mse_train = eval_metric(mse, skl, X_train, y_train)
skl_mse_valid = eval_metric(mse, skl, X_valid, y_valid)

print(f"MSE \t\tTrain\t\t\tValid \nMine : \t{mine_mse_train}\t{mine_mse_valid}\nSKL  : \t{skl_mse_train}\t{skl_mse_valid}")
# RMSE

mine_rmse_train = eval_metric(rmse, mine, X_train, y_train)
mine_rmse_valid = eval_metric(rmse, mine, X_valid, y_valid)

skl_rmse_train = eval_metric(rmse, skl, X_train, y_train)
skl_rmse_valid = eval_metric(rmse, skl, X_valid, y_valid)

print(f"RMSE \t\tTrain\t\t\tValid \nMine : \t{mine_rmse_train}\t{mine_rmse_valid}\nSKL  : \t{skl_rmse_train}\t{skl_rmse_valid}")