import warnings
warnings.filterwarnings('ignore')

#importing the Important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from matplotlib.pyplot import xticks
Cars_data = pd.read_csv('../input/car-data/CarPrice_Assignment.csv')
Cars_data.head()
Cars_data.shape
Cars_data.describe()
Cars_data.info()
#checking duplicates
sum(Cars_data.duplicated(subset = 'car_ID')) == 0
# No duplicate values
Cars_data.isnull().sum()*100/Cars_data.shape[0]
# There are no NULL values in the dataset, hence it is clean.
Cars_data.CarName.unique()
#Splitting company name from CarName column
CompanyName = Cars_data['CarName'].apply(lambda x : x.split(' ')[0])
Cars_data.insert(3,"CompanyName",CompanyName)
Cars_data.drop(['CarName'],axis=1,inplace=True)
Cars_data.head()
Cars_data.CompanyName.unique()
#There seems to be some spelling error in the CompanyName column.
Cars_data.CompanyName = Cars_data.CompanyName.str.lower()

def replace_name(a,b):
    Cars_data.CompanyName.replace(a,b,inplace=True)

replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')

Cars_data.CompanyName.unique()
#Checking for duplicates
Cars_data.loc[Cars_data.duplicated()]
sns.distplot(Cars_data['price'])


# Mean and median of price are significantly different.

# Price values are right-skewed, most cars are priced at the lower end (9000) of the price range.
sns.boxplot(Cars_data['price'])
plt.figure(figsize=(50, 6))

plt.subplot(1,3,1)
plt1 = Cars_data.CompanyName.value_counts().plot(kind='bar')
plt.title('Companies Histogram')
plt1.set(xlabel = 'Car company', ylabel='Frequency of company')
Cars_data.CompanyName.describe()
plt.figure(figsize=(50, 6))
plt.subplot(1,3,2)
plt1 = Cars_data.fueltype.value_counts().plot(kind='bar')
plt.title('Fuel Type Histogram')
plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of fuel type')
plt.figure(figsize=(50, 6))
plt.subplot(1,3,3)
plt1 = Cars_data.carbody.value_counts().plot(kind='bar')
plt.title('Car Type Histogram')
plt1.set(xlabel = 'Car Type', ylabel='Frequency of Car type')

#It seems that the symboling with 0 and 1 values have high number of rows (i.e. They are most sold.)
#The cars with -1 symboling seems to be high priced (as it makes sense too, insurance risk rating -1 is quite good). 
#But it seems that symboling with 3 value has the price range similar to -2 value. There is a dip in price at symboling 1.



plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Symboling Histogram')
sns.countplot(Cars_data.symboling, palette=("cubehelix"))

plt.subplot(1,2,2)
plt.title('Symboling vs Price')
sns.boxplot(x=Cars_data.symboling, y=Cars_data.price, palette=("cubehelix"))

plt.show()


plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Engine Type Histogram')
sns.countplot(Cars_data.enginetype, palette=("Blues_d"))

plt.subplot(1,2,2)
plt.title('Engine Type vs Price')
sns.boxplot(x=Cars_data.enginetype, y=Cars_data.price, palette=("PuBuGn"))

plt.show()

df = pd.DataFrame(Cars_data.groupby(['enginetype'])['price'].mean().sort_values(ascending = False))
df.plot.bar(figsize=(8,6))
plt.title('Engine Type vs Average Price')
plt.show()


#ohc Engine type seems to be most favored type.
#ohcv has the highest price range (While dohcv has only one row), ohc and ohcf have the low price range
plt.figure(figsize=(25, 6))

df = pd.DataFrame(Cars_data.groupby(['CompanyName'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Company Name vs Average Price')
plt.show()

df = pd.DataFrame(Cars_data.groupby(['fueltype'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Fuel Type vs Average Price')
plt.show()

df = pd.DataFrame(Cars_data.groupby(['carbody'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Car Type vs Average Price')
plt.show()



#Jaguar and Buick seem to have highest average price.
#diesel has higher average price than gas.
#hardtop and convertible have higher average price.
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16,9))
ax  = fig.gca(projection = "3d")

plot =  ax.scatter(Cars_data["price"],
           Cars_data["peakrpm"],
           Cars_data["horsepower"],
           linewidth=1,edgecolor ="k",
           c=Cars_data["price"],s=100,cmap="hot")

ax.set_xlabel("horsepower")
ax.set_ylabel("price")
ax.set_zlabel("peakrpm")

lab = fig.colorbar(plot,shrink=.5,aspect=5)
lab.set_label("price",fontsize = 15)

plt.title("3D plot for price, w.r.t peakrpm and horsepower",color="red")
plt.show()
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.title('Door Number Histogram')
sns.countplot(Cars_data.doornumber, palette=("plasma"))

plt.subplot(1,2,2)
plt.title('Door Number vs Price')
sns.boxplot(x=Cars_data.doornumber, y=Cars_data.price, palette=("plasma"))

plt.show()

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.title('Aspiration Histogram')
sns.countplot(Cars_data.aspiration, palette=("plasma"))

plt.subplot(1,2,2)
plt.title('Aspiration vs Price')
sns.boxplot(x=Cars_data.aspiration, y=Cars_data.price, palette=("plasma"))

plt.show()


#doornumber variable is not affacting the price much. There is no sugnificant difference between the categories in it.
#It seems aspiration with turbo have higher price range than the std(though it has some high values outside the whiskers.)
def plot_count(x,fig):
    plt.subplot(4,2,fig)
    plt.title(x+' Histogram')
    sns.countplot(Cars_data[x],palette=("magma"))
    plt.subplot(4,2,(fig+1))
    plt.title(x+' vs Price')
    sns.boxplot(x=Cars_data[x], y=Cars_data.price, palette=("magma"))
    
plt.figure(figsize=(15,20))

plot_count('enginelocation', 1)
plot_count('cylindernumber', 3)
plot_count('fuelsystem', 5)
plot_count('drivewheel', 7)

plt.tight_layout()

#Very few datapoints for enginelocation categories to make an inference.
#Most common number of cylinders are four, six and five. Though eight cylinders have the highest price range.
#mpfi and 2bbl are most common type of fuel systems. mpfi and idi having the highest price range. But there are few data for other categories to derive any meaningful inference
#A very significant difference in drivewheel category. Most high ranged cars seeme to prefer rwd drivewheel.
def scatter(x,fig):
    plt.subplot(5,2,fig)
    plt.scatter(Cars_data[x],Cars_data['price'])
    plt.title(x+' vs Price')
    plt.ylabel('Price')
    plt.xlabel(x)

plt.figure(figsize=(10,20))

scatter('carlength', 1)
scatter('carwidth', 2)
scatter('carheight', 3)
scatter('curbweight', 4)

plt.tight_layout()


#carwidth, carlength and curbweight seems to have a poitive correlation with price.
#carheight doesn't show any significant trend with price.
def pp(x,y,z):
    sns.pairplot(Cars_data, x_vars=[x,y,z], y_vars='price',size=4, aspect=1, kind='scatter')
    plt.show()

pp('enginesize', 'boreratio', 'stroke')
pp('compressionratio', 'horsepower', 'peakrpm')
pp('wheelbase', 'citympg', 'highwaympg')


#enginesize, boreratio, horsepower, wheelbase - seem to have a significant positive correlation with price.
#citympg, highwaympg - seem to have a significant negative correlation with price.
np.corrcoef(Cars_data['carlength'], Cars_data['carwidth'])[0, 1]
# Deriving new features, Fuel economy
Cars_data['fueleconomy'] = (0.55 * Cars_data['citympg']) + (0.45 * Cars_data['highwaympg'])
#Binning the Car Companies based on avg prices of each Company.
Cars_data['price'] = Cars_data['price'].astype('int')
temp = Cars_data.copy()
table = temp.groupby(['CompanyName'])['price'].mean()
temp = temp.merge(table.reset_index(), how='left',on='CompanyName')
bins = [0,10000,20000,40000]
cars_bin=['Budget','Medium','Highend']
Cars_data['carsrange'] = pd.cut(temp['price_y'],bins,right=False,labels=cars_bin)
Cars_data.head()
#Bivariate Analysis
plt.figure(figsize=(8,6))

plt.title('Fuel economy vs Price')
sns.scatterplot(x=Cars_data['fueleconomy'],y=Cars_data['price'],hue=Cars_data['drivewheel'])
plt.xlabel('Fuel Economy')
plt.ylabel('Price')

plt.show()
plt.tight_layout()

#fueleconomy has an obvios negative correlation with price and is significant.
plt.figure(figsize=(50, 6))

df = pd.DataFrame(Cars_data.groupby(['fuelsystem','drivewheel','carsrange'])['price'].mean().unstack(fill_value=0))
df.plot.bar()
plt.title('Car Range vs Average Price')
plt.show()
#significant variables after Visual analysis
cars_lr = Cars_data[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 
                    'fueleconomy', 'carlength','carwidth', 'carsrange']]
cars_lr.head()
sns.pairplot(cars_lr)
plt.show()
# Defining the map function
def dummies(x,df):
    temp = pd.get_dummies(df[x], drop_first = True)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df
# Applying the function to the cars_lr

cars_lr = dummies('fueltype',cars_lr)
cars_lr = dummies('aspiration',cars_lr)
cars_lr = dummies('carbody',cars_lr)
cars_lr = dummies('drivewheel',cars_lr)
cars_lr = dummies('enginetype',cars_lr)
cars_lr = dummies('cylindernumber',cars_lr)
cars_lr = dummies('carsrange',cars_lr)
cars_lr.head()
from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(cars_lr, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
#Correlation using heatmap
plt.figure(figsize = (30, 25))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()
#Dividing data into X and y variables
y_train = df_train.pop('price')
X_train = df_train
#Model Building
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
lm = LinearRegression()
lm.fit(X_train,y_train)
rfe = RFE(lm, 10)
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
X_train_rfe = X_train[X_train.columns[rfe.support_]]
X_train_rfe.head()
def build_model(X,y):
    X = sm.add_constant(X) #Adding the constant
    lm = sm.OLS(y,X).fit() # fitting the model
    print(lm.summary()) # model summary
    return X
    
def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)
#Model_one
X_train_new = build_model(X_train_rfe,y_train)
#p-vale of twelve and fueleconomy seems to be higher than the significance value of 0.05, 
#hence dropping it as it is insignificant in presence of other variables.
X_train_new = X_train_rfe.drop(["twelve","fueleconomy"], axis = 1)
#model_two
X_train_new = build_model(X_train_new,y_train)
#Calculating the Variance Inflation Factor
checkVIF(X_train_new)
#dropping curbweight because of high VIF value. (shows that curbweight has high multicollinearity.)
X_train_new = X_train_new.drop(["curbweight"], axis = 1)
#Model_three
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["sedan"], axis = 1)
#Model_four
X_train_new = build_model(X_train_new,y_train)
#dropping wagon because of high p-value.
X_train_new = X_train_new.drop(["wagon"], axis = 1)
#Model_Five
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)
#Dropping dohcv to see the changes in model statistics
#Model_six
X_train_new = X_train_new.drop(["dohcv"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)
#Residual Analysis of Model
lm = sm.OLS(y_train,X_train_new).fit()
y_train_price = lm.predict(X_train_new)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)   


#Error terms seem to be approximately normally distributed, so the assumption on the linear modeling seems to be fulfilled.
#Scaling the test set
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])
#Dividing into X and y
y_test = df_test.pop('price')
X_test = df_test
# Now let's use our model to make predictions.
X_train_new = X_train_new.drop('const',axis=1)
# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)
# Making predictions
y_pred = lm.predict(X_test_new)
#Evaluation of test via comparison of y_pred and y_test
from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)
#EVALUATION OF THE MODEL
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)   

print(lm.summary())
#Inference :
#R-sqaured and Adjusted R-squared (extent of fit) - 0.899 and 0.896 - 90% variance explained.
#F-stats and Prob(F-stats) (overall model fit) - 308.0 and 1.04e-67(approx. 0.0) - Model fir is significant and explained 90% variance is just not by chance.
#p-values - p-values for all the coefficients seem to be less than the significance level of 0.05. - meaning that all the predictors are statistically significant.