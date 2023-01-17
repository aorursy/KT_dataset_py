import pandas as pd
import numpy as np

data_red=pd.read_csv("../input/wine-data/winequality-red.csv",sep=",")
data_white=pd.read_csv("../input/wine-data/winequality-white.csv",sep=",")

print(data_red.head(10))
data_red["color"]="red"
data_white["color"]="white"
data_red_features = data_red.drop(["color"],axis=1)
data_white_features = data_white.drop(["color"],axis=1)

data_red_class = data_red.loc[:,"color"]
data_white_class = data_white.loc[:,"color"]

print(data_red_features.head(5))
print(data_red_class.head(5))
data_unity = pd.concat([data_red,data_white],ignore_index=True)

print(data_unity)

data_unity.loc[:,"color"] = [ 1 if each =="red" else 0 for each in data_unity.loc[:,"color"]]
print(data_unity)

import matplotlib.pyplot as plt
x1 = data_unity.iloc[0:100,5].values.reshape(-1,1) # Chlorides
y1 = data_unity.iloc[0:100,6].values.reshape(-1,1) # Total sulfur dioxide
plt.scatter(x1,y1)
plt.xlabel("Chlorides")
plt.ylabel("Total Sulfur Dioxide")
plt.show()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x1,y1)
y1_head = lr.predict(x1)

x1 = data_unity.iloc[0:100,5].values.reshape(-1,1) # Chlorides
y1 = data_unity.iloc[0:100,6].values.reshape(-1,1) # Total sulfur dioxide
plt.scatter(x1,y1)
plt.plot(x1,y1_head,color="red",label="linear")
plt.xlabel("Chlorides")
plt.ylabel("Total sulfur dioxide")
plt.show()
from sklearn.metrics import r2_score

print(r2_score(y1,y1_head)) 
z1 = data_unity.iloc[0:100,4:6] # Chlorides and free sulfur dioxide data 
print(z1.head(5))
mlr = LinearRegression()
mlr.fit(z1,y1)
prd=mlr.predict(z1)
print(prd[0]) # our prediction data of total sulfer dioxide
poly_data = pd.read_csv("../input/zombie-population-in-turkey/zombie-population.txt",sep = ",",header = None,names = ["days","zombie"])
x = poly_data.loc[:,"days"].values.reshape(-1,1)
y = poly_data.loc[:,"zombie"].values.reshape(-1,1)

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4) # x^n and degree=n=2. 
x_poly = poly_reg.fit_transform(x)
lr.fit(x_poly,y)
y_head= lr.predict(x_poly)

plt.scatter(x,y,color ="red")
plt.plot(x,y_head,color = "green",label="Pol")
plt.xlabel("Days")
plt.ylabel("Zombie population(Million)")
plt.show()
import numpy as np
xt = np.arange(1,11).reshape(-1,1) 
yt = np.ravel(np.arange(30,130,10).reshape(-1,1))

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(xt,yt)

xt_new = np.arange(min(xt),max(xt),0.01).reshape(-1,1)
yt_head = dtr.predict(xt_new)

plt.scatter(xt,yt,color="blue")
plt.plot(xt_new,yt_head,color="red")
plt.xlabel("xt")
plt.ylabel("yt")
plt.show()
xr = np.arange(1,11).reshape(-1,1)
yr = np.ravel(np.arange(30,130,10).reshape(-1,1))

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=10000,random_state=1) # n_estimators is the number of trees in the forest.
rfr.fit(xr,yr)

xr_new = np.arange(min(xr),max(xr),0.01).reshape(-1,1)
yr_head=rfr.predict(xr_new)

plt.plot(xr_new,yr_head,color = "red",label= "Random forest")
plt.scatter(xr,yr,color="blue")
plt.xlabel("xr")
plt.ylabel("yr")
plt.show()
from sklearn.metrics import r2_score

y_head_tree=dtr.predict(xt)
y_head_random=rfr.predict(xr)

print("Decision Tree r2 square score:",r2_score(yt,y_head_tree))
print("Random Forest r2 square score:",r2_score(yr,y_head_random))
x = data_unity.drop(["color"],axis=1)
y = data_unity.loc[:,"color"].values # Slipt the data two part : Features and class so we can predict them as a feautre
#and look to classifacation is true or not. 

x = (x-np.min(x))/(np.max(x)-np.min(x)).values # normalization for preclusioning data dominion

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2 , random_state=42) # test size=0.2 means 
#%20 of our data will be test data
print(x_test[0:3])
print(x_test.shape)

# We have to transforme to our test and train data for matrix calculate
x_train = x_train.T
x_test  = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x Train: ", x_train.shape)
print("x Test: ", x_test.shape)
print("y Train: ", y_train.shape)
print("y Test: ", y_test.shape)

def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head
def forward_backward_propagation(w,b,x_train,y_train): #Now we begin to train our data for prediction
    #forward propagation
    z = np.dot(w.T,x_train)+b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]

    #backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion): # Now, we reduce the cost function to find local minima
    cost_list = []
    cost_list2 = [] 
    index = []

    
    for i in range(number_of_iterarion):
        
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        #  
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
           

  
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

def predict(w,b,x_test): # we are choosing boundry( 0 and 1) for our data with a condition
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))

    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations): #magic happen here

    dimension =  x_train.shape[0]  
    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    print("test accuracy:%",(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 2000)

#Now we use sklearn libray .Much easier thanks god
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print(lr.predict(x_test.T[0:100]))
print("test accuracy :%",(lr.score(x_test.T,y_test.T)*100))
#creating two class with features 
r = data_unity[data_unity.loc[:,"color"] == 1]
w = data_unity[data_unity.loc[:,"color"] == 0]


plt.scatter(r.loc[:,"chlorides"],r.loc[:,"total sulfur dioxide"],color = "red",label="red",alpha = 0.3)
plt.scatter(w.loc[:,"chlorides"],w.loc[:,"total sulfur dioxide"],color = "green",label="white",alpha = 0.3)
plt.xlabel("Chlorides")
plt.ylabel("Total sulfur dioxide")
plt.legend()
plt.show()
x = data_unity.drop(["color"],axis=1)
y = data_unity.loc[:,"color"] # Slipt the data two part : Features and class so we can predict them as a feautre
#and look to classifacation is true or not. 

x = (x-np.min(x))/(np.max(x)-np.min(x)).values # normalization for preclusioning data dominion

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2 , random_state=42) # test size=0.2 means 
#%20 of our data will be test data
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2) # K value
knn.fit(x_train,y_train)
knn.predict(x_test)
print(" {} nn score {}".format(2,knn.score(x_test,y_test)))
# find finest k value 

list_k=[]
for i in range(1,20):
    knn1=KNeighborsClassifier(n_neighbors=i)
    knn1.fit(x_train,y_train)
    a=list_k.append(knn1.score(x_test,y_test))
    print("{} nn score {}".format(i,list_k[i-1]))

plt.plot(range(1,20),list_k)
plt.xlabel("Knn")
plt.ylabel("Accuracy")
plt.show() 