import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/mobile-price-classification/train.csv')
df.head()
df.describe()
# show the count and type of each colum
df.info()
# show if the data have any null values in it
df.isnull().sum()
# show every value in each column
for uniqu in df.columns:
    print(f"{uniqu:15}{df[uniqu].unique()}\n")
# show if the data is balanced or not with pie chart
plt.pie(df['price_range'].value_counts().values,labels=df['price_range'].unique(),autopct='%1.1f%%')
plt.title('price_range')
#plt.ylabel('wifi')
plt.show()
# show if the data is balanced or not with count plot
sns.countplot('price_range',data=df)
### We didn't use scatter plot because the data is categorical so we can use pie chart or bar chart or swarn plot
def newcircle(data,lab,title,fig=1):
    plt.subplot(1,4,fig)
    plt.pie(data.value_counts(),labels=lab,autopct='%1.1f%%')
    plt.title(title)
# how the touch screen availability in each category 
plt.figure(figsize=(15,15))
newcircle(df.loc[df['price_range']==3,'touch_screen'],['support touch screen','Not support touch screen'],'price_range',1)
newcircle(df.loc[df['price_range']==2,'touch_screen'],['support touch screen','Not support touch screen'],'price_range',2)
newcircle(df.loc[df['price_range']==1,'touch_screen'],['support touch screen','Not support touch screen'],'price_range',3)
newcircle(df.loc[df['price_range']==0,'touch_screen'],['support touch screen','Not support touch screen'],'price_range',4)
df[['touch_screen','price_range']].groupby(['price_range']).mean()
## Point plot to show relaction between ram and price
sns.pointplot(y="ram", x="price_range", data=df)
plt.figure(figsize=(20,20))
newcircle(df.loc[df['price_range']==0,'n_cores'],df['n_cores'].unique(),'price_range vs cores')
# how many cell phone support 3G
plt.figure(figsize=(20,20))
newcircle(df['three_g'],['support 3G','Not support 3G'],'3G')
sns.countplot('three_g',data=df)
X=df.drop('price_range',axis=1)
y=df['price_range']

from sklearn.model_selection import train_test_split
X_train, X_tes, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
pipe3 = Pipeline( [("RF", RandomForestClassifier())])
pipe3.fit(X_train, y_train)
print("Test score: {:.2f}".format(pipe3.score(X_tes, y_test)))
pr = pipe3.predict(X_tes)
print(classification_report(y_test,pr))
print("Train set score: {:.2f}".format(pipe3.score(X_train, y_train)))
from sklearn.linear_model import LogisticRegression
pipe2 = Pipeline( [("scaler", MinMaxScaler()),("lR", LogisticRegression(C=100))])
pipe2.fit(X_train, y_train)
print("Test score: {:.2f}".format(pipe2.score(X_tes, y_test)))
pr = pipe2.predict(X_tes)
print(classification_report(y_test,pr))
parameters = {'svm__kernel':('linear', 'rbf'), 'svm__C':[0.001, 0.01, 0.1, 1, 10, 100],'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
svc = SVC()
pipe = Pipeline( [("svm", svc)])
grid = GridSearchCV(pipe, param_grid=parameters,cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_tes, y_test)))
print("Best parameters: {}".format(grid.best_params_))
print("Train set score: {:.2f}".format(grid.score(X_train, y_train)))
print(classification_report(y_test,grid.predict(X_tes)))
test_data = pd.read_csv('../input/mobile-price-classification/test.csv')
test_data.head()
test_data.drop('id',axis=1,inplace=True)
test_data.head()
predicted_price=grid.predict(test_data)

predicted_price
test_data['price_range']=predicted_price
test_data.head()
