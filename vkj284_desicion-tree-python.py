import pandas as pd
df = pd.read_csv('../input/playercsv/player.csv')
df.head()
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df.outlook = label.fit_transform(df.outlook)
df.temperature = label.fit_transform(df.temperature)
df.humidity = label.fit_transform(df.humidity)
df.wind = label.fit_transform(df.wind)
df.playgolf = label.fit_transform(df.playgolf)
x = df[['outlook','temperature','humidity','wind']]
print("input:\n",x)
y = df.playgolf
print("output:\n",y)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(xtrain,ytrain)
ypred = DTC.predict(xtest)
NDS = pd.DataFrame()
NDS['outlook'] = xtest['outlook'].copy()
NDS['temperature'] = xtest['temperature'].copy()
NDS['humidity'] = xtest['humidity'].copy()
NDS['wind'] = xtest['wind'].copy()
NDS['ytest'] = ytest.copy()
NDS['ypred'] = ypred.copy()
NDS
from sklearn import tree
dot_data = tree.export_graphviz(DTC,out_file=None,filled = True ,rounded = True)
import graphviz
graph2 = graphviz.Source(dot_data)
graph2