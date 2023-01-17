import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
labelencoder = LabelEncoder()
df = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv") #veri setini import ettik
df.isnull().sum() # or data.isnull().sum().sum()
df.info()
df.rename(columns = {"race/ethnicity" : "race_ethnicity","parental level of education":"parental_level_of_education", 
                      "test preparation course":"test_preparation_course","math score":"math_score","reading score":"reading_score",
                      "writing score":"writing_score", },inplace=True)
df.head()
df.gender = pd.Categorical(df.gender)
df.race_ethnicity = pd.Categorical(df.race_ethnicity)
df.parental_level_of_education = pd.Categorical(df.parental_level_of_education)
df.lunch = pd.Categorical(df.lunch)
df.test_preparation_course  = pd.Categorical(df.test_preparation_course)
df.info()
df2 = pd.DataFrame(df["gender"])
df2['gender_cat'] = labelencoder.fit_transform(df2["gender"])
df2
df2["gender"].append(df2["gender_cat"]).unique()
enc = OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(df2[['gender_cat']]).toarray())
enc_df = enc_df.rename(columns = {0 : "female",1: "male"})
df = df.join(enc_df)
df.head()
df2 = pd.DataFrame(df["race_ethnicity"])
df2['race_ethnicity_cat'] = labelencoder.fit_transform(df2["race_ethnicity"])
df2
df2["race_ethnicity"].append(df2["race_ethnicity_cat"]).unique()
enc_df = pd.DataFrame(enc.fit_transform(df2[['race_ethnicity_cat']]).toarray())
enc_df = enc_df.rename(columns = {0 : "group_A",1: "group_B", 2 : "group_C", 3 : "group_D", 4 : "group_E"})
df = df.join(enc_df)
df.head()
df2 = pd.DataFrame(df["parental_level_of_education"])
df2['parental_level_of_education_cat'] = labelencoder.fit_transform(df2["parental_level_of_education"])
df2
df2["parental_level_of_education"].append(df2["parental_level_of_education_cat"]).unique()
enc_df = pd.DataFrame(enc.fit_transform(df2[['parental_level_of_education_cat']]).toarray())
enc_df = enc_df.rename(columns = {0 : "associate's_degree",1: "bachelor's_degree", 2 : "high_school", 3 : "master's_degree", 4 : "some_college",
                                 5 : "some_high_school"})
df = df.join(enc_df)
df.head()
df2 = pd.DataFrame(df["lunch"])
df2['lunch_cat'] = labelencoder.fit_transform(df2["lunch"])
df2
df2["lunch"].append(df2["lunch_cat"]).unique()
enc_df = pd.DataFrame(enc.fit_transform(df2[['lunch_cat']]).toarray())
enc_df = enc_df.rename(columns = {0 : "free-reduced",1: "standard"})
df = df.join(enc_df)
df.head()
df2 = pd.DataFrame(df["test_preparation_course"])
df2['test_preparation_course_cat'] = labelencoder.fit_transform(df2["test_preparation_course"])
df2
df2["test_preparation_course"].append(df2["test_preparation_course_cat"]).unique()
enc_df = pd.DataFrame(enc.fit_transform(df2[['test_preparation_course_cat']]).toarray())
enc_df = enc_df.rename(columns = {0 : "completed",1: "none"})
df = df.join(enc_df)
df.head()
df.describe().T
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(3)
x = df.iloc[:, 6:7].values
y = df.iloc[:, 5].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 34)
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)
print("R score: {0}".format(round(linear_regressor.score(x_train, y_train),2)))
print("Intercept: {0}".format(round(linear_regressor.intercept_),))
#pd.DataFrame({'feature':x.columns, 'coef':linear_regressor.coef_})
y_predictions = linear_regressor.predict(x_test)
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, linear_regressor.predict(x_train), color = "blue")
plt.title("PREDICTING MATH SCORE WITH READING SCORE")
plt.xlabel("READING SCORE")
plt.ylabel("MATH SCORE")
#plt.savefig('basic_linear_regression.png')
plt.show()
mae = mean_absolute_error(linear_regressor.predict(x_test), y_test)
mse = mean_squared_error(linear_regressor.predict(x_test), y_test)
rmse = np.sqrt(mse)

print('Mean Absolute Error (MAE): %.2f' % mae)
print('Mean Squared Error (MSE): %.2f' % mse)
print('Root Mean Squared Error (RMSE): %.2f' % rmse)
X = df.iloc[:, 7:8].values
Y = df.iloc[:, 5].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 34)
linear_regressor.fit(X_train, Y_train)
print("R score: {0}".format(round(linear_regressor.score(X_train, Y_train),2)))
print("Intercept: {0}".format(round(linear_regressor.intercept_),))
#pd.DataFrame({'feature':X.columns, 'coef':linear_regressor.coef_})
plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, linear_regressor.predict(X_train), color = "blue")
plt.title("PREDICTING MATH SCORE WITH WRITING SCORE")
plt.xlabel("WRITING SCORE")
plt.ylabel("MATH SCORE")
#plt.savefig('basic_linear_regression.png')
plt.show()
mae = mean_absolute_error(linear_regressor.predict(X_test), Y_test)
mse = mean_squared_error(linear_regressor.predict(X_test), Y_test)
rmse = np.sqrt(mse)

print('Mean Absolute Error (MAE): %.2f' % mae)
print('Mean Squared Error (MSE): %.2f' % mse)
print('Root Mean Squared Error (RMSE): %.2f' % rmse)