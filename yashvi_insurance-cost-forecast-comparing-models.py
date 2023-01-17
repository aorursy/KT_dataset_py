import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
data=pd.read_csv('../input/insurance/insurance.csv')
data.head()
data.iloc[:,[0,2,6]].describe()
sns.distplot(data['age'])
sns.boxplot(y='age',data=data,color='green')
sns.scatterplot(x="age", y="bmi", hue='sex',data=data,color='red')
sns.boxplot(x='sex',y='age',data=data)
f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.sex == 'male')]["age"],color='b',ax=ax)
ax.set_title('Distribution of ages of male')

ax=f.add_subplot(122)
sns.distplot(data[(data.sex == 'female')]['age'],color='r',ax=ax)
ax.set_title('Distribution of ages of female')
sns.boxplot(x='smoker',y='age',data=data)
f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.smoker == 'yes')]["age"],color='#b0b0b0',ax=ax)
ax.set_title('Distribution of ages of smoker')

ax=f.add_subplot(122)
sns.distplot(data[(data.smoker == 'no')]['age'],color='#333ed6',ax=ax)
ax.set_title('Distribution of ages of non smoker')
sns.catplot(x="smoker", kind="count",hue = 'sex',palette='GnBu',data=data)
sns.boxplot(x='smoker',y='charges',palette='viridis',data=data)
f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.smoker == 'yes')]["charges"],color='#b0b0b0',ax=ax)
ax.set_title('Distribution of charges of smoker')

ax=f.add_subplot(122)
sns.distplot(data[(data.smoker == 'no')]['charges'],color='#333ed6',ax=ax)
ax.set_title('Distribution of charges of non smoker')
sns.boxplot(x='smoker',y='charges',data=data[(data.age>=18)&(data.age<=22)])
sns.scatterplot(x="age", y="charges", data=data[data.smoker=='yes'],color='purple')
g = sns.jointplot(x="age", y="charges", data=data[data.smoker=='yes'], kind="kde", color="b")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("age", "charges");
sns.jointplot(x="age", y="charges", data=data[data.smoker=='yes'], kind="kde");
sns.scatterplot(x="age", y="charges", data=data[data.smoker=='no'],color='#82113a')
g = sns.jointplot(x="age", y="charges", data=data[data.smoker=='no'], kind="kde", color="#82113a")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("age", "charges");
sns.jointplot(x="age", y="charges", data=data[data.smoker=='no'],color='#82113a', kind="kde");
sns.boxplot(x='sex',y='bmi',palette='viridis',data=data)
f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.sex == 'male')]["bmi"],color='b',ax=ax)
ax.set_title('Distribution of bmi of male')

ax=f.add_subplot(122)
sns.distplot(data[(data.sex == 'female')]['bmi'],color='r',ax=ax)
ax.set_title('Distribution of bmi of female')
f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.smoker == 'yes')]["bmi"],color='#b0b0b0',ax=ax)
ax.set_title('Distribution of bmi of smoker')

ax=f.add_subplot(122)
sns.distplot(data[(data.smoker == 'no')]['bmi'],color='#333ed6',ax=ax)
ax.set_title('Distribution of bmi of non smoker')
sns.catplot(x="region", kind="count",palette='viridis',data=data)
sns.catplot(x="region", kind="count",hue = 'sex',palette='viridis',data=data)
sns.lmplot(x="bmi", y="charges",data=data);
sns.lmplot(x="bmi", y="charges", hue="smoker", data=data, palette="Set1")
sns.lmplot(x="bmi", y="charges", hue="smoker", col="sex", data=data)
sns.lmplot(x="bmi", y="charges", col="children", data=data,aspect=.5)
g = sns.jointplot(x="bmi", y="charges", data=data, kind="kde", color="#4837cc")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("bmi", "charges")
sns.catplot(x="children", kind="count",palette='rainbow',data=data)
sns.lmplot(x="children", y="charges",data=data)
sns.lmplot(x="children", y="charges", hue='smoker',data=data)
f, ax = plt.subplots(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr)
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error

#sex
le = LabelEncoder()
le.fit(data.sex.drop_duplicates()) 
data.sex = le.transform(data.sex)
# smoker or not
le.fit(data.smoker.drop_duplicates()) 
data.smoker = le.transform(data.smoker)
#region
le.fit(data.region.drop_duplicates()) 
data.region = le.transform(data.region)
data.head()
x = data.drop(['charges','region'], axis = 1)
y = data.charges

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)
lreg = linear_model.LinearRegression()
lreg.fit(x_train,y_train)
y_train_pred = lreg.predict(x_train)
y_test_pred = lreg.predict(x_test)
lreg.score(x_test,y_test)
degree=2
polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
polyreg.fit(x_train,y_train)
y_train_pred = polyreg.predict(x_train)
y_test_pred = polyreg.predict(x_test)
polyreg.score(x_test,y_test)
dt_regressor = DecisionTreeRegressor(random_state=0)
cross_val_score(dt_regressor,x_train, y_train, cv=10).mean()
Rf = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
Rf.fit(x_train,y_train)
Rf_train_pred = Rf.predict(x_train)
Rf_test_pred = Rf.predict(x_test)


r2_score(y_test,Rf_test_pred)