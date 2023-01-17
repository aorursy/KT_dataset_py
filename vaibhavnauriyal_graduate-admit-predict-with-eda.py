import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

dataset=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
dataset.columns
dataset.dtypes
dataset['Admit_rate']=dataset['Chance of Admit ']
dataset=dataset.drop('Chance of Admit ', axis=1)
sns.set(style="whitegrid")
sns.distplot(dataset['Admit_rate'])
sns.boxplot(dataset['Admit_rate'])
sns.lineplot(x=dataset['GRE Score'], y=dataset['Admit_rate'])
sns.lineplot(x=dataset['TOEFL Score'], y=dataset['Admit_rate'])
sns.barplot(x=dataset['SOP'], y=dataset['Admit_rate'])
dataset['LOR']=dataset['LOR ']
dataset=dataset.drop('LOR ', axis=1)
sns.barplot(x=dataset['LOR'], y=dataset['Admit_rate'])
sns.lineplot(x=dataset['CGPA'], y=dataset['Admit_rate'])
sns.barplot(x=dataset['Research'], y=dataset['Admit_rate'])
sns.barplot(x=dataset['University Rating'], y=dataset['Admit_rate'])
sns.lineplot(x=dataset['GRE Score'], y=dataset['Admit_rate'], hue=dataset['Research'])
palette = sns.color_palette("mako_r", 5)
sns.lineplot(x=dataset['GRE Score'], y=dataset['Admit_rate'], hue=dataset['University Rating'],palette=palette, legend="full")
palette = sns.color_palette("mako_r", 9)
sns.lineplot(x=dataset['GRE Score'], y=dataset['Admit_rate'], hue=dataset['SOP'],  palette=palette,legend="full")
dataset['CGPA_cat']=0
temp = [dataset]
for data in temp:
    data.loc[dataset['CGPA']<=8,'CGPA_cat']=3,
    data.loc[(dataset['CGPA']>8) & (dataset['CGPA']<=9),'CGPA_cat']=2,
    data.loc[(dataset['CGPA']>9) & (dataset['CGPA']<=10),'CGPA_cat']=1
    
palette = sns.color_palette("mako_r", 3)
sns.lineplot(x=dataset['GRE Score'], y=dataset['Admit_rate'], hue=dataset['CGPA_cat'], palette=palette, legend="full")
sns.lineplot(x=dataset['TOEFL Score'], y=dataset['Admit_rate'], hue=dataset['Research'])
palette = sns.color_palette("mako_r", 5)
sns.lineplot(x=dataset['TOEFL Score'], y=dataset['Admit_rate'], hue=dataset['University Rating'],palette=palette, legend="full")
palette = sns.color_palette("mako_r", 9)
sns.lineplot(x=dataset['TOEFL Score'], y=dataset['Admit_rate'], hue=dataset['SOP'],  palette=palette,legend="full")
palette = sns.color_palette("mako_r", 3)
sns.lineplot(x=dataset['TOEFL Score'], y=dataset['Admit_rate'], hue=dataset['CGPA_cat'],palette=palette, legend="full")
dataset.isnull().sum()
dataset2=dataset.drop("Serial No.", axis=1)
corr=dataset2.corr()
sns.heatmap(corr, cmap="YlGnBu")
corr = dataset2.corr()
fig, ax = plt.subplots(figsize=(8, 8))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)
plt.title('Graduate Admissions - Features Correlations')
plt.show()
print (corr['Admit_rate'].sort_values(ascending=False)[:10], '\n') #top 15 values


num = [f for f in dataset2.columns if ((dataset2.dtypes[f] != 'object')& (dataset2.dtypes[f]!='bool'))]

nd = pd.melt(dataset2, value_vars = num)
n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value')
target=dataset['Admit_rate']
drop=['Serial No.', 'Admit_rate','CGPA_cat']
train = dataset.drop(drop, axis=1)
#Now we will split the dataset in the ratio of 75:25 for train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.25, random_state = 0)
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

model = XGBRegressor(n_estimators=1000,learning_rate=0.009,n_jobs=-1)
model.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))
    

y_pred = model.predict(X_test)
print("XGBoost Regressor MSE on testing set: ", mean_squared_error(y_test, y_pred))
print("XGBoost Regressor RMSE on testing set: ", rmse(y_test, y_pred))

print("XGBoost Regressor Score: {}".format(100*max(0,1-(rmse(y_test,y_pred)))))
sns.scatterplot(y_test,y_pred)
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Chance of Admit Test and predicted data")
plt.legend()
plt.show()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
  
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
from sklearn.linear_model import Lasso
best_alpha = 0.0099

regr = Lasso(alpha=best_alpha, max_iter=50000)
regr.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))
y_pred = regr.predict(X_test)
print("Lasso Regressor MSE on testing set: ", mean_squared_error(y_test, y_pred))
print("Lasso Regressor RMSE on testing set: ", rmse(y_test, y_pred))
print("Lasso Regressor Score: {}".format(100*max(0,1-(rmse(y_test,y_pred)))))
sns.scatterplot(y_test,y_pred)
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Chance of Admit Test and predicted data")
plt.legend()
plt.show()
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))

y_pred = lr.predict(X_test)
print("Linear Regressor MSE on testing set: ", mean_squared_error(y_test, y_pred))
print("Linear Regressor RMSE on testing set: ", rmse(y_test, y_pred))
print("Linear Regressor Score: {}".format(100*max(0,1-(rmse(y_test,y_pred)))))
sns.scatterplot(y_test,y_pred)
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Chance of Admit Test and predicted data")
plt.legend()
plt.show()
gre = int(input("Enter GRE Score - "))
tfl = int(input("Enter TOEFL Score - "))
ur = int(input("Enter Your University Ratings(1 being lowest and 5 being highest) - "))
sop = int(input("Enter Your SOP(s) Ratings(1 being lowest and 5 being highest) - "))
cgpa = float(input("Enter Your CGPA(1-10) - "))
rp = int(input("Enter number of research publications under your name - "))
lor = int(input("Enter number of Letter of Recommendations - "))
check = {'GRE Score': gre, 'TOEFL Score':tfl,'University Rating':ur,'SOP':sop,'CGPA':cgpa,
         'Research':rp,'LOR':lor}
df = pd.DataFrame(check,columns = ['GRE Score','TOEFL Score','University Rating','SOP','CGPA','Research','LOR'],
                 index=[1])
chances = (model.predict(df))*100
chances = round(chances[0])
print('Your chances of getting an admit - {}%'.format(chances))