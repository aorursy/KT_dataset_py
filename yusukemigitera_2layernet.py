import numpy as np

import pandas as pd

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.head(10)
def missing_table(df): 

        null_val = df.isnull().sum()

        percent = 100 * df.isnull().sum()/len(df)

        kesson_table = pd.concat([null_val, percent], axis=1)

        kesson_table_ren_columns = kesson_table.rename(

        columns = {0 : 'missing', 1 : '%'})

        return kesson_table_ren_columns
train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1

train["Age"] = train["Age"].fillna(train["Age"].median())

test["Age"] = test["Age"].fillna(test["Age"].median())

test["Fare"] = test["Fare"].fillna(test["Fare"].median())
missing_table(train)
missing_table(test)
def sigmoid(x):

    return 1 / (1 + np.exp(-x.astype(np.float64)))    



def softmax(x):

    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策

    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)



def cross_entropy_error(y, t):

    if y.ndim == 1:

        t = t.reshape(1, t.size)

        y = y.reshape(1, y.size)

        

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換

    if t.size == y.size:

        t = t.argmax(axis=1)

             

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size



def numerical_gradient(f, x):

    h = 1e-4 # 0.0001

    grad = np.zeros_like(x)

    

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:

        idx = it.multi_index

        tmp_val = x[idx]

        x[idx] = tmp_val + h

        fxh1 = f(x) # f(x+h)

        

        x[idx] = tmp_val - h 

        fxh2 = f(x) # f(x-h)

        grad[idx] = (fxh1 - fxh2) / (2*h)

        

        x[idx] = tmp_val # 値を元に戻す

        it.iternext()   

        

    return grad
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        # 重みの初期化

        self.params = {}

        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)

        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)

        self.params['b2'] = np.zeros(output_size)



    def predict(self, x):

        W1, W2 = self.params['W1'], self.params['W2']

        b1, b2 = self.params['b1'], self.params['b2']

    

        a1 = np.dot(x, W1) + b1

        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2

        y = softmax(a2)

        

        return y

        

    # x:入力データ, t:教師データ

    def loss(self, x, t):

        y = self.predict(x)

        

        return cross_entropy_error(y, t)



    # x:入力データ, t:教師データ

    def numerical_gradient(self, x, t):

        loss_W = lambda W: self.loss(x, t)

        

        grads = {}

        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])

        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])

        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])

        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        

        return grads
import sys



x_train = np.array([train["Pclass"],train["Sex"],train["Age"],train["Fare"]]).T

t_train = np.array(train["Survived"])



iters_num = 100  # 繰り返しの回数を適宜設定する

train_size = x_train.shape[0]

batch_size = 100

learning_rate = 0.5



network = TwoLayerNet(input_size=4, hidden_size=30, output_size=2)

train_loss_list = []



for i in range(iters_num):

    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]

    t_batch = t_train[batch_mask]

    

    # 勾配の計算

    grad = network.numerical_gradient(x_batch, t_batch)

    #grad = network.gradient(x_batch, t_batch)

    

    # パラメータの更新

    for key in ('W1', 'b1', 'W2', 'b2'):

        network.params[key] -= learning_rate * grad[key]

    

    loss = network.loss(x_batch, t_batch)

    train_loss_list.append(loss)

    

    sys.stdout.write("\rcomplete {}/{}\n".format(i, iters_num))



print(train_loss_list)
x_test = np.array([test["Pclass"],test["Sex"],test["Age"],test["Fare"]]).T

y = network.predict(x_test)

z = []

for p in y:

    if p[0] > p[1]:

        z.append(0)

    else:

        z.append(1)

Z = np.array(z)

print(Z.shape)

print(Z)
# PassengerIdを取得

PassengerId = np.array(test["PassengerId"]).astype(int)

# Z(予測データ）とPassengerIdをデータフレームへ落とし込む

my_solution = pd.DataFrame(Z, PassengerId, columns = ["Survived"])

# my_tree_one.csvとして書き出し

my_solution.to_csv("my_prediction.csv", index_label = ["PassengerId"])