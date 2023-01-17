import numpy as np
import pandas as pd
df=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
df.head()
df.tail()
df.sample(5)
df[['Research']].sample(5)
#DataFrame Object
df[['CGPA']].head()
#Series
df.CGPA.head()
df['CGPA'].head()
df[df['TOEFL Score']>=120]
df1= df.reindex(columns=['Serial No.', 'TOEFL Score', 'University Rating', 'CGPA', 'Chance of Admit'])
df1[df1['TOEFL Score']>=120][['Serial No.','University Rating','CGPA','Chance of Admit']].head()
df.describe()
df.info()
X = df.iloc[:, 0:8].values

y = df.iloc[:, 8].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train
y_train
X_test
y_test
# Feature Scaling of training and test set

from sklearn.preprocessing import StandardScaler

sc_X_train = StandardScaler()

X_train = sc_X_train.fit_transform(X_train)



sc_X_test = StandardScaler()

X_test = sc_X_test.fit_transform(X_test)
#Fitting the model

from sklearn.ensemble import RandomForestRegressor



from sklearn.metrics import mean_absolute_error

# compare MAE with differing values of max_leaf_nodes

def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):

    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test,y_pred)

    return(mae)
for max_leaf_nodes in [5, 50, 100, 300, 500, 700, 800, 850]:

    my_mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
y_test
regressor = RandomForestRegressor(n_estimators=700, random_state=0)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred
regressorScore = regressor.score(X_test,y_test)
regressorScore
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor



from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor

from sklearn.metrics import mean_squared_error



Mod = [['DecisionTree :',DecisionTreeRegressor()],

           ['Linear Regression :', LinearRegression()],

           ['RandomForest :',RandomForestRegressor()],

           ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],

           ['SVM :', SVR()],

           ['AdaBoostClassifier :', AdaBoostRegressor()],

           ['GradientBoostingClassifier: ', GradientBoostingRegressor()],

           ['Lasso: ', Lasso()],

           ['Ridge: ', Ridge()],

           ['BayesianRidge: ', BayesianRidge()],

           ['ElasticNet: ', ElasticNet()],

           ['HuberRegressor: ', HuberRegressor()]]

print("Results...")

for n1,m1 in Mod:

    m1 = m1

    m1.fit(X_train, y_train)

    predictions = m1.predict(X_test)

    print(n1, (np.sqrt(mean_squared_error(y_test, predictions))))



print("Accuracy...")  

for n1,m1 in Mod:

    m1 = m1

    a=m1.score(X_test,y_test)

    print("Accuracy of ",n1,a )