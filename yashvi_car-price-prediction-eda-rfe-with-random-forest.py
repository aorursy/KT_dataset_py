import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
sns.set(rc={'figure.figsize':(8,8)})
data=pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')
data.head()
data=data.drop(['car_ID'],axis=1)
data['CarName'] = data['CarName'].str.split(' ',expand=True)
data['CarName'] = data['CarName'].replace({'maxda': 'mazda', 'nissan': 'Nissan', 'porcshce': 'porsche', 'toyouta': 'toyota', 
                            'vokswagen': 'volkswagen', 'vw': 'volkswagen'})
data['symboling']=data['symboling'].astype('str')
categorical_cols=data.select_dtypes(include=['object']).columns
data[categorical_cols].head(2)
numerical_cols=data.select_dtypes(exclude=['object']).columns
data[numerical_cols].head(2)
data.describe()
df=pd.DataFrame(data['CarName'].value_counts()).reset_index().rename(columns={'index':'car_name','CarName': 'count'})
plot = sns.barplot(y='car_name',x='count',data=df)
plot=plt.setp(plot.get_xticklabels(), rotation=80)
df=pd.DataFrame(data['fueltype'].value_counts())
plot = df.plot.pie(y='fueltype', figsize=(5, 5))
sns.distplot(data['price'],kde=True)
f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.fueltype== 'gas')]["price"],color='b',ax=ax)
ax.set_title('Distribution of price of gas vehicles')

ax=f.add_subplot(122)
sns.distplot(data[(data.fueltype == 'diesel')]['price'],color='r',ax=ax)
ax.set_title('Distribution of ages of diesel vehicles')
sns.boxplot(x = 'fueltype', y = 'price', data = data,palette='Pastel2')
df=pd.DataFrame(data['aspiration'].value_counts())
plot = df.plot.pie(y='aspiration', figsize=(5, 5))
f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
plot=sns.distplot(data[(data.aspiration== 'turbo')]["price"],color='#ca91eb',ax=ax)
ax.set_title('Price distribution of Turbo aspiration vehicles')

ax=f.add_subplot(122)
plot=sns.distplot(data[(data.aspiration == 'std')]['price'],color='#eb6426',ax=ax)
ax.set_title('Price distribution of Std aspiration vehicles')

sns.boxplot(x = 'aspiration', y = 'price', data = data,palette='Pastel1')
df=pd.DataFrame(data['symboling'].value_counts()).reset_index().rename(columns={'index':'symboling','symboling':'count'})
sns.barplot(x='symboling',y='count',data=df)
sns.boxplot(x = 'symboling', y = 'price', data = data,palette='Pastel1')
df=pd.DataFrame(data['doornumber'].value_counts())
plot = df.plot.pie(y='doornumber', figsize=(5, 5))
f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
plot=sns.distplot(data[(data.doornumber== 'two')]["price"],color='#ca91eb',ax=ax)
ax.set_title('Price distribution of cars having two doors')

ax=f.add_subplot(122)
plot=sns.distplot(data[(data.doornumber == 'four')]['price'],color='#eb6426',ax=ax)
ax.set_title('Price distribution of cars having four doors')

sns.boxplot(x = 'doornumber', y = 'price', data = data,palette='Accent')
df=pd.DataFrame(data['carbody'].value_counts())
plot = df.plot.pie(y='carbody', figsize=(8, 8))
sns.boxplot(x = 'carbody', y = 'price', data = data,palette='Accent')
df=pd.DataFrame(data['drivewheel'].value_counts())
plot = df.plot.pie(y='drivewheel', figsize=(8, 8))
sns.boxplot(x = 'drivewheel', y = 'price', data = data,palette='Accent')
df=pd.DataFrame(data['enginelocation'].value_counts())
plot = df.plot.pie(y='enginelocation', figsize=(8, 8))
df=pd.DataFrame(data['enginetype'].value_counts())
plot = df.plot.pie(y='enginetype', figsize=(8, 8))
sns.boxplot(x = 'enginetype', y = 'price', data = data,palette='Accent')
df=pd.DataFrame(data['cylindernumber'].value_counts())
plot = df.plot.pie(y='cylindernumber', figsize=(8, 8))
sns.boxplot(x = 'cylindernumber', y = 'price', data = data,palette='Accent')
df=pd.DataFrame(data['fuelsystem'].value_counts()).reset_index().rename(columns={'index':'fuelsystem','fuelsystem':'count'})
sns.barplot(x='fuelsystem',y='count',data=df)
sns.boxplot(x = 'fuelsystem', y = 'price', data = data,palette='gist_rainbow')
sns.scatterplot(x="wheelbase", y="price", data=data,color='purple')
g = sns.jointplot(x="wheelbase", y="price", data=data, kind="kde", color="b")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("wheel base", "price");
sns.scatterplot(x="carlength", y="price", data=data,color='b')
g = sns.jointplot(x="carlength", y="price", data=data, kind="kde", color="pink")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("car length", "price");
sns.scatterplot(x="carwidth", y="price", data=data,color='b')
g = sns.jointplot(x="carwidth", y="price", data=data, kind="kde", color="pink")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("car width", "price");
sns.scatterplot(x="carlength", y="carwidth", data=data,color='b')
g = sns.jointplot(x="carwidth", y="carlength", data=data, kind="kde", color="pink")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("car width", "car length");
sns.scatterplot(x="curbweight", y="price", data=data,color='b')
g = sns.jointplot(x="curbweight", y="price", data=data, kind="kde", color="b")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("curbweight", "price");
sns.scatterplot(x="enginesize", y="price", data=data,color='b')
g = sns.jointplot(x="enginesize", y="price", data=data, kind="kde", color="b")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("enginesize", "price");
sns.scatterplot(x="boreratio", y="price", data=data,color='b')
sns.scatterplot(x="stroke", y="price", data=data,color='b')
sns.scatterplot(x="compressionratio", y="price", data=data,color='b')
sns.scatterplot(x="horsepower", y="price", data=data,color='b')
g = sns.jointplot(x="horsepower", y="price", data=data, kind="kde", color="b")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("horsepower", "price");
sns.scatterplot(x="peakrpm", y="price", data=data,color='r')
sns.scatterplot(x="citympg", y="price", data=data,color='b')
g = sns.jointplot(x="citympg", y="price", data=data, kind="kde", color="b")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("citympg", "price");
sns.scatterplot(x="highwaympg", y="price", data=data,color='b')
g = sns.jointplot(x="highwaympg", y="price", data=data, kind="kde", color="b")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("highwaympg", "price");
ax = sns.pairplot(data[numerical_cols])
data[numerical_cols].corr()
sns.heatmap(data[numerical_cols].corr())
col=['wheelbase','carlength','carwidth','curbweight','price']
sns.pairplot(data[col])
sns.heatmap(data[col].corr())
col=['carlength','highwaympg','curbweight','price']
sns.pairplot(data[col])
sns.heatmap(data[col].corr())
col=['carwidth','curbweight','enginesize','price']
sns.pairplot(data[col])
sns.heatmap(data[col].corr())
col=['curbweight','enginesize','horsepower','highwaympg','price']
sns.pairplot(data[col])
sns.heatmap(data[col].corr())
col=['horsepower','citympg','highwaympg','price']
sns.pairplot(data[col])
sns.heatmap(data[col].corr())
sns.pairplot(data[['horsepower','price','carbody']], hue="carbody");
fig,axes = plt.subplots(4,4,figsize=(18,15))
for seg,col in enumerate(numerical_cols[:len(numerical_cols)-1]):
    
    x,y = seg//4,seg%4
    sns.regplot(x=col, y='price' ,data=data,ax=axes[x][y],color='r')

X=data[numerical_cols].drop('price',axis=1)
y=data['price']
X = data.apply(lambda col: preprocessing.LabelEncoder().fit_transform(col))
X=X.drop(['CarName','price'],axis=1)
y=data['price']

# Create the RFE object and rank each pixel
clf_rf_3 = RandomForestRegressor()      
rfe = RFE(estimator=clf_rf_3, n_features_to_select=15, step=1)
rfe = rfe.fit(X, y)
print('Chosen best 15 feature by rfe:',X.columns[rfe.support_])

features=list(X.columns[rfe.support_])
x = X[features]
y = data.price
x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)
lreg = linear_model.LinearRegression()
lreg.fit(x_train,y_train)
y_train_pred = lreg.predict(x_train)
y_test_pred = lreg.predict(x_test)
lreg.score(x_test,y_test)
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(x_train,y_train)
y_train_pred = dt_regressor.predict(x_train)
y_test_pred = dt_regressor.predict(x_test)
dt_regressor.score(x_test,y_test)
Rf = RandomForestRegressor(n_estimators = 15,
                              criterion = 'mse',
                              random_state = 20,
                              n_jobs = -1)
Rf.fit(x_train,y_train)
Rf_train_pred = Rf.predict(x_train)
Rf_test_pred = Rf.predict(x_test)


r2_score(y_test,Rf_test_pred)