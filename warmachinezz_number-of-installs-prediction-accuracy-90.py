import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
Application_data=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
Application_data.head()
Application_data.columns
Application_data.describe()
#Checking whether there are null values in the Ratings column
nullcheck_ratings=pd.isnull(Application_data["Rating"])
Application_data[nullcheck_ratings]
#Replacing the NaN values with the mean rating value
Application_data["Rating"].fillna(value=Application_data["Rating"].mean(),inplace=True)
Application_data["Rating"]

# Checking the unique values in the Rating column,we find there is an inconsistent value of 19.
Application_data["Rating"].unique()
# Replacing the inconsistent value with the mean value of ratings
Application_data["Rating"].replace(19.,4.1,inplace=True)
# Checking the unique values of the number of reviews column, we find there are no unrelated values.
len(Application_data["Reviews"].unique())
# Checking the Null values of the number of reviews column, we find there are no null values.
nullcheck_reviews=pd.isnull(Application_data["Reviews"])
Application_data[nullcheck_reviews]
# Checking for any special character that might prevent numeric conversion, 3.0M is replaced with its real value to make the data consistent.
Application_data["Reviews"].replace("3.0M","3000000",inplace=True)
# Finally converting the datatype of Reviews column from Object type(String) to Numeric type(float or int)
Application_data["Reviews"]=pd.to_numeric(Application_data["Reviews"])
# Checking for the unique values of the Size column, it is observed it has values appended with M,k and "Varies with device"
Application_data["Size"].unique()
# Replacing the "Varies with device" field with NaN entry, so that later on these can be replaced with mean values.
Application_data['Size'].replace('Varies with device', np.nan, inplace = True )
Application_data['Size'].replace('1,000+', np.nan, inplace = True )

# Checking for null values which we will find, since in the above line we have added few null values.
nullcheck_size=pd.isnull(Application_data["Size"])
Application_data[nullcheck_size]

Application_data.Size = (Application_data.Size.replace(r'[kM]+$','', regex=True).astype(float) *
                         Application_data.Size.str.extract(r'[\d\.]+([kM]+)', expand=False).fillna(1).replace(['k','M'], [10**3, 10**6]).astype(int))
# Finally replacing the NaN values with the mean value.
Application_data["Size"].fillna(value="21516530",inplace=True)
# After removing the special characters, lets convert it to numeric data type for finding the mean value.
Application_data["Size"]=pd.to_numeric(Application_data["Size"])
# Checking the unique values of the column Installs, we observe that there is a type called "free", which is inconsistent and non numeric, so it should be replaced.
Application_data["Installs"].unique()
# Removing the "+" symbol to make the column numeric.
Application_data["Installs"]=Application_data["Installs"].map(lambda x: x.rstrip('+'))
# Removing the "," from the digits to make it easier.
Application_data["Installs"]=Application_data["Installs"].str.replace(",","")
# There was no null entries found in this column
nullcheck_installs=pd.isnull(Application_data["Installs"])
Application_data[nullcheck_installs]
# Replacing the inconsistent label value with the mean value of the column.
Application_data["Installs"].replace("Free","15462910",inplace=True)
# Converting the Datatype to the numeric type for analysis
Application_data["Installs"]=pd.to_numeric(Application_data["Installs"])
# Checking for the unique values, we found nan and 0 which should be replaced with Free.
Application_data["Type"].unique()
# Replacing 0 with Free
Application_data["Type"].replace("0","Free",inplace=True)

# Filling the missing values with Free, since most of the applications are free on Google play.
Application_data["Type"].fillna(value="Free",inplace=True)
# Addding the dummy columns for this, so that it can contribute to our model.
dummy_type=pd.get_dummies(Application_data["Type"])
#Concatenating the dummy columns with the main dataframe.
Application_data=pd.concat([Application_data,dummy_type],axis=1)
# Finally dropping the type column.
Application_data.drop(["Type"],axis=1,inplace=True)
Application_data.head()
# By checking the unique values we observe that "Everyone" is an inconsistent value that should be removed.
Application_data["Price"].unique()
# Removing the dollar symbol
Application_data["Price"]=Application_data["Price"].map(lambda x: x.lstrip('$'))
# Removing the non essential row value.
Application_data.drop(Application_data[Application_data["Price"] == "Everyone"].index, inplace=True)
# By checking there were no null values found
nullcheck_Prices=pd.isnull(Application_data["Price"])
Application_data[nullcheck_Prices]
# Finally converting to numeric type for analysis
Application_data["Price"]=pd.to_numeric(Application_data["Price"])
# Checking the unique values, we found 
Application_data["Category"].unique()
Application_data["Category"].replace("1.9","MISCELLANEOUS",inplace=True)
# Checking for null values, there were no null values found for this column
nullcheck=pd.isnull(Application_data["Category"])
Application_data[nullcheck]
# Importing the required library
from sklearn.preprocessing import LabelEncoder
# Instantiating the encoder
labelencoder2 = LabelEncoder()
#Encoding the Ctegory column using scikit learn
Application_data['Categories_encoded'] = labelencoder2.fit_transform(Application_data['Category'])
# finally dropping the type column, since it is already splitted.
Application_data.drop(["Category"],axis=1,inplace=True)
Application_data.head()
# Checking for unique values
Application_data["Content Rating"].unique()
# Null check for Content Rating
nullcheck_contentrating=pd.isnull(Application_data["Content Rating"])
Application_data[nullcheck_contentrating]
# importing the required package
from sklearn.preprocessing import LabelEncoder
#instantiating the encoder
labelencoder = LabelEncoder()
# encoding the column
Application_data['Content_Rating_encoded'] = labelencoder.fit_transform(Application_data['Content Rating'])
# finally removing the content ratig column after encoding
Application_data.drop(["Content Rating"],axis=1,inplace=True)
Application_data.head()
# Checking the datatypes of the columns to ensure that we have successfully gathered all the numerical columns.
Application_data.dtypes
# Finding the mean of all the numerical columns
Application_data.mean()
sns.pairplot(Application_data)
colorassigned=Application_data["Rating"]
fig = px.histogram(Application_data, x="Rating", marginal="rug",
                   hover_data=Application_data.columns,nbins=30,color=colorassigned)
fig.show()
fig = px.histogram(Application_data, x="Reviews", marginal="rug",
                   hover_data=Application_data.columns,nbins=30)
fig.show()
colorassigned=Application_data["Size"]
fig = px.histogram(Application_data, x="Size", marginal="rug",
                   hover_data=Application_data.columns,nbins=30,color=colorassigned)
fig.show()
colorassigned=Application_data["Installs"]
fig = px.histogram(Application_data, x="Installs", marginal="rug",
                   hover_data=Application_data.columns,nbins=30,color=colorassigned)
fig.show()
colorassigned=Application_data["Price"]
fig = px.histogram(Application_data, x="Price", marginal="rug",
                   hover_data=Application_data.columns,nbins=30,color=colorassigned)
fig.show()
# Calculating the Correlation and plotting the heatmap to know the relations.
cors=Application_data.corr()
fig = px.imshow(cors,labels=dict(color="Pearson Correlation"), x=['Rating', 'Reviews', 'Size', 'Installs', 'Price','Paid','Free','Content_Rating_encoded','Categories_encoded'],
                y=['Rating', 'Reviews', 'Size','Installs','Price','Paid','Free','Content_Rating_encoded','Categories_encoded'])
fig.show()
# Plotting scatter plot with a line of fit between Installs and Reviews, these two have the highest Correlation between them.
from scipy.stats import pearsonr 
corryu,_ =pearsonr(Application_data["Installs"],Application_data["Reviews"])
colorassigned=Application_data["Reviews"]
fig = px.scatter(Application_data, x="Installs", y="Reviews",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)
# Plotting scatter plot with a line of fit between Rating and Reviews, these two have very less correlation between them. 
from scipy.stats import pearsonr 
corryu,_ =pearsonr(Application_data["Rating"],Application_data["Reviews"])
colorassigned=Application_data["Reviews"]
fig = px.scatter(Application_data, x="Rating", y="Reviews",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)
# Plotting scatter plot with a line of fit between Size and Reviews, these two have very less correlation between them. 
from scipy.stats import pearsonr 
corryu,_ =pearsonr(Application_data["Size"],Application_data["Reviews"])
colorassigned=Application_data["Reviews"]
fig = px.scatter(Application_data, x="Size", y="Reviews",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)
from scipy.stats import pearsonr 
corryu,_ =pearsonr(Application_data["Installs"],Application_data["Categories_encoded"])
colorassigned=Application_data["Categories_encoded"]
fig = px.scatter(Application_data, x="Installs", y="Categories_encoded",trendline="ols",color=colorassigned)
fig.show()
print("Pearson Correlation: %.3f" % corryu)
print("P-value: %.8f" % _)
# Splitting the target variable and the feature matrix
X=Application_data[["Reviews","Size","Rating","Price","Paid","Free","Categories_encoded","Content_Rating_encoded"]]
y=Application_data["Installs"]
# importing train test set
from sklearn.model_selection import train_test_split
# splitting the training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# importing linear regressor
from sklearn.linear_model import LinearRegression
# Instantiating linear regressor
lm=LinearRegression()
# Fitting the model
lm.fit(X_train,y_train)
# making predictions on the test set
predictions=lm.predict(X_test)
# displaying predictions
predictions
# Accuracy score for Linear regressor
linearregressionscore=lm.score(X_test,y_test)
linearregressionscore
# The coefficient for Linear regressor per feature.
lm.coef_
# Importing the metrics
from sklearn import metrics
# Mean absolute error on test data
metrics.mean_absolute_error(y_test,predictions)
# Mean squared error on test data
metrics.mean_squared_error(y_test,predictions)
# Root mean squared error on test data
rmelinear=np.sqrt(metrics.mean_absolute_error(y_test,predictions))
rmelinear
# Defining the feature matrix and the target variable
X=Application_data[["Reviews","Size","Rating","Price","Paid","Free","Categories_encoded","Content_Rating_encoded"]]
y=Application_data["Installs"]
# Importing the train test split
from sklearn.model_selection import train_test_split
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Importing the regressor
from sklearn.tree import DecisionTreeRegressor
# Instantiating the regressor
decisiontreereg=DecisionTreeRegressor()
# Fitting the model
decisiontreereg.fit(X_train,y_train)
# Gettting the predicted values
y_prediction=decisiontreereg.predict(X_test)
# The accuracy score for decision tree regressor
decisiontreescore=decisiontreereg.score(X_test,y_test)
decisiontreescore
from sklearn import metrics
# Mean absolute error
metrics.mean_absolute_error(y_test,y_prediction)
# Root mean square error
rmetree=np.sqrt(metrics.mean_absolute_error(y_test,y_prediction))
rmetree
# Separating the feature matrix and target variable
X=Application_data[["Reviews","Size","Rating","Price","Paid","Free","Categories_encoded","Content_Rating_encoded"]]
y=Application_data["Installs"]
# Importing the train test split
from sklearn.model_selection import train_test_split
# Splitting the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Importing the random forest regressor
from sklearn.ensemble import RandomForestRegressor
# Instantiating with giving the value of number of sub trees to be created.
Randomforestreg=RandomForestRegressor(n_estimators = 100,n_jobs = -1,oob_score = True, bootstrap = True,random_state=42)
# fitting the model
Randomforestreg.fit(X_train,y_train)
# Predicting the number of installs
y_prediction_randomforest=Randomforestreg.predict(X_test)
# Using barplot from seaborn to show importance of features in sorted manner.
feature_imp=pd.DataFrame(sorted(zip(Randomforestreg.feature_importances_,Application_data[["Reviews","Size","Rating","Price","Paid","Free","Categories_encoded","Content_Rating_encoded"]])),columns=["Significance","Features"])
fig=plt.figure(figsize=(6,6))
sns.barplot(x="Significance",y="Features",data=feature_imp.sort_values(by="Significance",ascending=False),dodge=False)
plt.title("Important features for predicting the number of installs")
plt.tight_layout()
plt.show()

from sklearn.metrics import r2_score,mean_squared_error
# The performance of random forest.
print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(Randomforestreg.score(X_train, y_train), 
                                                                                             Randomforestreg.oob_score_,
                                                                                             Randomforestreg.score(X_test, y_test)))
# Accuracy score for random forest
randomforestscore=Randomforestreg.score(X_test,y_test)
randomforestscore
# Importing the performance metrics
from sklearn import metrics
# Mean absolute error
metrics.mean_absolute_error(y_test,y_prediction_randomforest)
# Root mean squared error
rmerandom=np.sqrt(metrics.mean_absolute_error(y_test,y_prediction_randomforest))
rmerandom
# Creating the dataframe that has accuracy score and root mean squared error for all the 3 models.
dict={"Linear Regressor":[linearregressionscore,rmelinear],"DecisionTree Regressor":[decisiontreescore,rmetree],"RandomForest Regressor":[randomforestscore,rmerandom]}
df_comparison_models=pd.DataFrame(dict,["Score","Root Mean Square Error"])
df_comparison_models.head()
# Plotting the accuracy of all the 3 models
%matplotlib inline
model_accuracy = pd.Series(data=[linearregressionscore,decisiontreescore,randomforestscore], 
        index=['Linear_Regressor','DecisionTree Regressor','RandomForest Regressor'])
fig= plt.figure(figsize=(8,8))
model_accuracy.sort_values().plot.barh()
plt.title('Model Accuracy')
# Plotting the Root Mean Squared Error comaparison
%matplotlib inline
model_accuracy = pd.Series(data=[rmelinear,rmetree,rmerandom], 
        index=['Linear_Regressor','DecisionTree Regressor','RandomForest Regressor'])
fig= plt.figure(figsize=(8,8))
model_accuracy.sort_values().plot.barh()
plt.title('Model Root Mean Squared Error')