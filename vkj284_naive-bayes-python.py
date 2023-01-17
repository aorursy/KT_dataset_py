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
y = df.playgolf
print("input:\n",x)
print("output:\n",y)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(xtrain,ytrain)
ypred = gnb.predict(xtest)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(ytest,ypred)
aas = accuracy_score(ytest,ypred)
print("confusion matrix")
print(cm)
print("\n accuracy score")
print(aas)
gnb.predict([[2,0,0,1]])