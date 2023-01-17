import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.colors



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,mean_squared_error



from tqdm import tqdm_notebook
test_url = "../input/contest1likeordislike/test.csv"

df_test = pd.read_csv(test_url)

#df_test.head
train_url = "../input/contest1likeordislike/train.csv"

df_train = pd.read_csv(train_url)

#df_train.head
x_train = df_train[['Brand','Capacity','Internal Memory','RAM','Bezel-less display','Rating Count','Fingerprint Sensor','Resolution']]

y_train = df_train[['Rating']]

x_test = df_test[['Brand','Capacity','Internal Memory','RAM','Bezel-less display','Rating Count','Fingerprint Sensor','Resolution']]



# Making a single dataframe for further computation by merging train and test datasets row-wise

df = pd.concat([x_train, x_test], ignore_index=True)

df.shape

#df
# "BRAND"



#df['Brand'].isnull().sum() 

#This gives us 1 as output.



#x_train['Brand'].isnull().sum() 

# There is a null value in 'Brand' column of the train dataset 



#x_test['Brand'].isnull().sum()



#Finding the index of the row containing the null value:

#x_train[x_train['Brand'].isnull()].index.tolist() 

#This gives us [5] as output.



#Since the train column has one missing 'Brand' value row, therefore it must be dropped,

df = df['Brand'].dropna()



x_train = x_train.drop(x_train.index[5])

df = pd.concat([x_train, x_test], ignore_index=True)



df.shape
# "CAPACITY"



df['Capacity'].fillna(32,inplace=True)
#df['Capacity'].isnull().sum() 



df['Capacity'] = df['Internal Memory'].str.split().str[0]

df['Capacity'] = df['Capacity'].astype(float)
# "INTERNAL MEMORY"



df['Internal Memory'].fillna(16,inplace=True)
#Visualising each row of dataframe manually in order to get good results

#pd.set_option("max_rows",None)

#df



#Also from the above visualisation, for Internal Memory column,values in MB and even in KB are there...

#so converting them in GB



for r in range (len(df['Internal Memory'].str.split().str[1])):

    if r == "MB":

        df['Internal Memory'] = df['Internal Memory'].str.split().str[0]/1000

    elif r == "KB":

        df['Internal Memory'] = df['Internal Memory'].str.split().str[0]/1000000

    else:

        df['Internal Memory'] = df['Internal Memory'].str.split().str[0]

        

df['Internal Memory'] = df['Internal Memory'].astype(float)       
# "RAM"



df['RAM'].fillna(1,inplace=True)
#df['RAM'].isnull().sum() 



for r in range (len(df['RAM'].str.split().str[1])):

    if r == "MB":

        df['RAM'] = df['RAM'].str.split().str[0]/1000

    else:

        df['RAM'] = df['RAM'].str.split().str[0]

        

df['RAM'] = df['RAM'].astype(float)       
# "Bezel-less display"



df['Bezel-less display'].fillna('no',inplace=True)
# "Rating Count"



mean = df['Rating Count'].mean(skipna=None)

round(mean)
for i in range(len(df['Rating Count'])):

    c = df['Rating Count'][i]

    if c>=34418:

        df['Rating Count'][i]="High"

    else:

        df['Rating Count'][i]="Low"
# "Fingerprint Sensor"



df['Fingerprint Sensor'].fillna('no',inplace=True)
# "Resolution"



df['Resolution'] = df['Resolution'].str.split().str[0]
df['Resolution'].fillna('2',inplace=True)
pd.set_option("max_rows",None)

df
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
df['Brand'] = LE.fit_transform(df['Brand'])

df['Capacity'] = LE.fit_transform(df['Capacity'])

df['Internal Memory'] = LE.fit_transform(df['Internal Memory'])

df['RAM'] = LE.fit_transform(df['RAM'])

df['Bezel-less display'] = LE.fit_transform(df['Bezel-less display'])

df['Rating Count'] = LE.fit_transform(df['Rating Count'])

df['Fingerprint Sensor'] = LE.fit_transform(df['Fingerprint Sensor'])

df['Resolution'] = LE.fit_transform(df['Resolution'])
y_train = y_train.drop(y_train.index[5])
def binarize(X,num):

    Y=[]

    if type(X) is pd.core.frame.DataFrame:

        X=np.array(X)

    for x in X:

        if(x>=num):

            Y.append(1)

        else:

            Y.append(0)

    return np.array(Y)   
Y = binarize(y_train,4)

Y = np.reshape((Y),newshape=(354,1)) 

Y
#Splitting of the prepared dataset into Train and Test dataframes for model evaluation.



x_train = df.iloc[0:354,:]

x_test = df.iloc[354:,:]
#Scaling 



from sklearn.preprocessing import StandardScaler

SS = StandardScaler()
x_train = SS.fit_transform(x_train)

x_test  = SS.transform(x_test)
X_train,X_test,Y_train,Y_test = train_test_split(x_train,Y, test_size = 1/4, random_state = 1, stratify=Y)
#Splitting the data into train and val



X_train, X_val, Y_train, Y_val = train_test_split(x_train,Y, test_size = 1/4, random_state = 1, stratify=Y)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, X_val.shape, Y_val.shape, X_test.shape)
class FFNeuralNetwork:

  

  def __init__(self, inputs, hidden=[2]):

    

    self.x = inputs

    self.y = 1

    self.h = len(hidden)

    self.sizes = [self.x] + hidden + [self.y]

    

    self.W = {}

    self.B = {}

    for i in range(self.h+1):

      self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])

      self.B[i+1] = np.zeros((1, self.sizes[i+1]))

  

  def sigmoid(self, x):

    return 1.0/(1.0 + np.exp(-x))



  def grad_sigmoid(self, x):

    return x*(1-x) 

  

  def fwd_prop(self, x):

    self.A = {}

    self.H = {}

    self.H[0] = x.reshape(1, -1)

    for i in range(self.h+1):

      self.A[i+1] = np.matmul(self.H[i], self.W[i+1]) + self.B[i+1]

      self.H[i+1] = self.sigmoid(self.A[i+1])

    return self.H[self.h+1]

    

  def back_prop(self, x, y):

    self.fwd_prop(x)

    self.dW = {}

    self.dB = {}

    self.dH = {}

    self.dA = {}

    L = self.h + 1

    self.dA[L] = (self.H[L] - y)

    for i in range(L, 0, -1):

      self.dW[i] = np.matmul(self.H[i-1].T, self.dA[i])

      self.dB[i] = self.dA[i]

      self.dH[i-1] = np.matmul(self.dA[i], self.W[i].T)

      self.dA[i-1] = np.multiply(self.dH[i-1], self.grad_sigmoid(self.H[i-1]))

    

  def fit(self, X, Y, epochs, learning_rate, initialise=True, display_loss=False):

    

    if initialise:

      for i in range(self.h+1):

        self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])

        self.B[i+1] = np.zeros((1, self.sizes[i+1]))

      

    if display_loss:

      loss = {}

    

    for epoch in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):

      dW = {}

      dB = {}

      for i in range(self.h+1):

        dW[i+1] = np.zeros((self.sizes[i], self.sizes[i+1]))

        dB[i+1] = np.zeros((1, self.sizes[i+1]))

      for x, y in zip(X, Y):

        self.back_prop(x, y)

        for i in range(self.h+1):

          dW[i+1] += self.dW[i+1]

          dB[i+1] += self.dB[i+1]

        

      n = X.shape[1]

    

      for i in range(self.h+1):

        self.W[i+1] -= learning_rate*dW[i+1]/n

        self.B[i+1] -= learning_rate*dB[i+1]/n

      

      if display_loss:

        Y_pred = self.predict(X)

        loss[epoch] = mean_squared_error(Y_pred, Y)

    

    if display_loss:

      plt.plot(list(loss.values()))

      plt.xlabel('Epochs')

      plt.ylabel('MSE Loss')

      plt.show()

      

  def predict(self, X):

    Y_pred = []

    for x in X:

        y_pred = self.fwd_prop(x)

        Y_pred.append(y_pred)

    return np.array(Y_pred).squeeze()
#Object of the model



ffnn = FFNeuralNetwork(8, [4, 2])

ffnn.fit(X_test, Y_test, epochs=1000, learning_rate=.04, display_loss=True)



Y_pred_test = ffnn.predict(X_test)

Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()

Y_pred_val = ffnn.predict(X_val)

Y_pred_binarised_val = (Y_pred_val >= 0.5).astype("int").ravel()

accuracy_test = accuracy_score(Y_pred_binarised_test, Y_test)

accuracy_val = accuracy_score(Y_pred_binarised_val, Y_val)
print("Test accuracy", accuracy_test)
print("Validation accuracy", accuracy_val)
Output = Y_pred_binarised_test

Output = pd.DataFrame(Output)

sample = pd.read_csv("../input/contest1likeordislike/test.csv")

sample = sample["PhoneId"]

Output = pd.concat([sample, Output], axis = 1)

Output.to_csv("output_01.csv", header = ["PhoneId", "Class"], index = False)