import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

pd.set_option('display.max_columns', None)

import warnings

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import tools

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

from sklearn import datasets, linear_model

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error as MSE

from sklearn.model_selection import GridSearchCV

from sklearn.svm import LinearSVR

from sklearn.svm import SVR

from xgboost.sklearn import XGBRegressor

from sklearn.tree import export_graphviz



warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 9999

pd.options.display.float_format = '{:20,.2f}'.format

from IPython.display import Image

from IPython.core.display import HTML 
#Loading mathematic data set

data_mat = pd.read_csv('../input/student-mat.csv')

#Loading Portuguese data set

data_por = pd.read_csv('../input/student-por.csv')

data_mat.info()
data_mat.head()
data_por.head()
#Validate nulls on mathematic dataset

data_mat.isnull().any()
#Validate nulls on Portuguese dataset

data_por.isnull().any()
#Display number of students study mathematic according to gender

sns.catplot(x="sex", kind="count",palette="magma", data=data_mat, height = 6)

plt.title("Gender of students : F - female,M - male")
#Display number of students study Portuguese according to gender.

sns.catplot(x="sex", kind="count",palette="magma", data=data_por, height = 6)

plt.title("Gender of students : F - female,M - male")
#Display distribution of math students according to age.

ages_mat = data_mat["age"].value_counts()

labels_mat = (np.array(ages_mat.index))

sizes_mat = (np.array((ages_mat / ages_mat.sum())*100))



ages_por = data_por["age"].value_counts()

labels_por = (np.array(ages_por.index))

sizes_por = (np.array((ages_por / ages_por.sum())*100))



trace = go.Pie(labels=labels_mat, values=sizes_mat)

layout = go.Layout(title="Аge of students")

dat = [trace]

fig = go.Figure(data=dat, layout=layout)

py.iplot(fig, filename="age")

#Display distribution of Portuguese students according to age.

trace = go.Pie(labels=labels_por, values=labels_por)

layout = go.Layout(title="Аge of students")

dat = [trace]

fig = go.Figure(data=dat, layout=layout)

py.iplot(fig, filename="age")
#Display how many hours per week math students spend on their studies.

data_mat['st_time'] = np.nan

df = [data_mat]



for col in df:

    col.loc[col['studytime'] == 1 , 'st_time'] = '< 2 hours'

    col.loc[col['studytime'] == 2 , 'st_time'] = '2 to 5 hours'

    col.loc[col['studytime'] == 3, 'st_time'] = '5 to 10 hours'

    col.loc[col['studytime'] == 4, 'st_time'] = '> 10 hours'  

 

labels = data_mat["st_time"].unique().tolist()

amount = data_mat["st_time"].value_counts().tolist()



colors = ["red", "blue", "grey", "yellow"]



trace = go.Pie(labels=labels, values=amount,

               hoverinfo='label+percent', textinfo='value', 

               textfont=dict(size=20),

               marker=dict(colors=colors, 

                           line=dict(color='#000000', width=2)))

dt = [trace]

layout = go.Layout(title="Study time -Math ")



fig = go.Figure(data=dt, layout=layout)

iplot(fig, filename='pie')
#Display how many hours per week Portugese students spend on their studies.

data_por['st_time'] = np.nan

df = [data_por]



for col in df:

    col.loc[col['studytime'] == 1 , 'st_time'] = '< 2 hours'

    col.loc[col['studytime'] == 2 , 'st_time'] = '2 to 5 hours'

    col.loc[col['studytime'] == 3, 'st_time'] = '5 to 10 hours'

    col.loc[col['studytime'] == 4, 'st_time'] = '> 10 hours'  

 

labels = data_por["st_time"].unique().tolist()

amount = data_por["st_time"].value_counts().tolist()



colors = ["red", "blue", "grey", "yellow"]



trace = go.Pie(labels=labels, values=amount, hoverinfo='label+percent', textinfo='value', 

               textfont=dict(size=20), marker=dict(colors=colors, line=dict(color='#000000', width=2)))



dt = [trace]

layout = go.Layout(title="Study time - Portugese")



fig = go.Figure(data=dt, layout=layout)

iplot(fig, filename='pie')
#Display math students travling time urban vs rural.

sns.catplot(x="address", kind="count",hue = "traveltime",palette="brg", data=data_mat, height = 6)

plt.title("Students address for Mathematics courses: U - urban City, R - rural non City")



#Display Portugese students travling time urban vs rural.

sns.catplot(x="address", kind="count",hue = "traveltime",palette="brg", data=data_por, height = 6)

plt.title("Students address for Portugese courses: U - urban City, R - rural non City")
#Categories math & Portuguese students according to their grade.

# 15-20 	Excellent

# 10-15 	Good

# 0-9       Poor   



data_mat['Category_Grade'] = 'na'

data_mat.loc[(data_mat.G3  >= 15) & (data_mat.G3 <= 20), 'Category_Grade'] = 'Excellent' 

data_mat.loc[(data_mat.G3  >= 10) & (data_mat.G3 <= 14), 'Category_Grade'] = 'GOOD' 

data_mat.loc[(data_mat.G3  >= 0) & (data_mat.G3 <= 9), 'Category_Grade'] = 'POOR'



data_por['Category_Grade'] = 'na'

data_por.loc[(data_por.G3  >= 15) & (data_por.G3 <= 20), 'Category_Grade'] = 'Excellent' 

data_por.loc[(data_por.G3  >= 10) & (data_por.G3 <= 14), 'Category_Grade'] = 'GOOD' 

data_por.loc[(data_por.G3  >= 0) & (data_por.G3 <= 9), 'Category_Grade'] = 'POOR' 





data_mat.head(5)
#Display final grade category in math and Portuguese according to category

plt.figure(figsize=(8,6))

sns.countplot(data_mat.Category_Grade, order=["POOR","GOOD","Excellent"], palette='Set1')

plt.title('Final Grade - Mathematics',fontsize=20)

plt.xlabel('Final Grade', fontsize=16)

plt.ylabel('Number of Student', fontsize=16)



plt.figure(figsize=(8,6))

sns.countplot(data_por.Category_Grade, order=["POOR","GOOD","Excellent"], palette='Set1')

plt.title('Final Grade - Portuguese',fontsize=20)

plt.xlabel('Final Grade', fontsize=16)

plt.ylabel('Number of Student', fontsize=16)
# Display crrelation between features for math and Portugase students

plt.figure(figsize=(10,10))

sns.heatmap(data_mat.corr(),annot = True,fmt = ".2f",cbar = True)

plt.title('Crrelation - Mathematics',fontsize=20)

plt.xticks(rotation=90)

plt.yticks(rotation = 0)



plt.figure(figsize=(10,10))

sns.heatmap(data_por.corr(),annot = True,fmt = ".2f",cbar = True)

plt.title('Crrelation - Portuguese',fontsize=20)

plt.xticks(rotation=90)

plt.yticks(rotation = 0)
# Display grade distibution for math and Portuguese according G1,G2 & G3 semesters.

fig = plt.figure(figsize=(14,5))

plt.style.use('seaborn-white')

ax1 = plt.subplot(121)

plt.hist([data_mat['G1'], data_mat['G2'], data_mat['G3']], label=['G1', 'G2', 'G3'], color=['#48D1CC', '#FF7F50', '#778899' ], alpha=0.8)

plt.legend(fontsize=14)

plt.xlabel('Grade', fontsize=18)

plt.ylabel('Frequency', fontsize=18)

plt.title('Mathematics Grades', fontsize=20)

ax1.spines['top'].set_visible(False)

ax1.spines['right'].set_visible(False)

plt.ylim(0,220)



ax2 = plt.subplot(122)

plt.hist([data_por['G1'], data_por['G2'], data_por['G3']], label=['G1', 'G2', 'G3'], color=['#48D1CC', '#FF7F50', '#778899' ], alpha=0.8)

plt.legend(fontsize=14)

plt.xlabel('Grade', fontsize=18)

plt.ylabel('Frequency', fontsize=18)

plt.title('Portuguese Grades', fontsize=20)

ax2.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)

plt.ylim(0,220)



plt.show()
# Display distribution of math and Portuguese students absences

fig = plt.figure(figsize=(14,10))



ax1 = plt.subplot(221)

sns.countplot(data_mat['absences'], color='#48D1CC')

plt.title('Absences count in Math', fontsize=14)

plt.xlabel('number of absences', fontsize=12)

ax1.spines['top'].set_visible(False)

ax1.spines['right'].set_visible(False)

plt.xlim((0,32))



ax2 = plt.subplot(222)

sns.countplot(data_por['absences'], color='#FF7F50')

plt.title('Absences count in Portuguese', fontsize=14)

plt.xlabel('number of absences', fontsize=12)

ax2.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)

plt.xlim((0,32))



ax3 = plt.subplot(223)

sns.regplot(data_mat['absences'], data_mat['G3'], x_estimator=np.mean, color='#48D1CC')

plt.title('Math: G3 vs absences', fontsize=14)

plt.xlabel('Absences: number of absences', fontsize=12)

ax3.spines['top'].set_visible(False)

ax3.spines['right'].set_visible(False)

plt.xlim((0,32))



ax4 = plt.subplot(224)

sns.regplot(data_por['absences'], data_por['G3'], x_estimator=np.mean, color='#FF7F50')

plt.title('Portuguese: G3 vs absences', fontsize=14)

plt.xlabel('Absences: number of absences', fontsize=12)

ax4.spines['top'].set_visible(False)

ax4.spines['right'].set_visible(False)

plt.xlim((0,32))



plt.tight_layout()
# Preprocessing final data while columns that contain Yes/No values will be converted into binary values and categories columns 

# will be enumerated.



math_final = data_mat.copy()

math_final = math_final.drop(['G1', 'G2','st_time','Category_Grade'], axis=1)

# Convert dummy variables values into 0/1.

math_final.school = math_final['school'].replace(['GP', 'MS'], [1,0])

math_final.sex = math_final['sex'].replace(['F','M'],[1,0])

math_final.address = math_final['address'].replace(['U','R'], [1,0])

math_final.famsize = math_final['famsize'].replace(['LE3','GT3'], [1,0])

math_final.Pstatus = math_final['Pstatus'].replace(['T','A'], [1,0])

math_final.schoolsup = math_final['schoolsup'].replace(['yes','no'],[1,0])

math_final.famsup = math_final['famsup'].replace(['yes','no'],[1,0])

math_final.activities = math_final['activities'].replace(['yes','no'],[1,0])

math_final.nursery = math_final['nursery'].replace(['yes','no'],[1,0])

math_final.higher = math_final['higher'].replace(['yes','no'],[1,0])

math_final.internet = math_final['internet'].replace(['yes','no'],[1,0])

math_final.romantic = math_final['romantic'].replace(['yes','no'],[1,0])

math_final.paid = math_final['paid'].replace(['yes','no'],[1,0])



norminal_vars = ['Fjob', 'Mjob', 'reason','guardian','Medu','Fedu','traveltime','studytime']

math_final = pd.get_dummies(math_final, columns= norminal_vars, drop_first=True)

math_final.head()
# Split data into train and test.

X = math_final.drop(['G3'], axis=1)

y = math_final['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Run cross val score on the fpllowing chosen models:

#    1.) LinearRegression

#    2.) DecisionTreeRegressor

#    3.) linear_model.Lasso

#    4.) GradientBoostingRegressor

#    5.) RandomForestRegressor

#    6.) SVR

#    7.) XGBRegressor





#Create the model according to give model name    

def bulid_model(model_name):

        model = model_name()

        return model



#Run the given model and prints the score and std

def run_cross_val_score(model, X, y, cv):

    #Running the scoring with negative mean squared error 

    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')

    cv_scores = np.sqrt(abs(cv_scores)) # Convert the given score into RMSE

    print("CV Score cv =", cv, cv_scores, "\n\nMean of cv scores: ", np.mean(cv_scores),"\n")

    print("STD =", cv_scores.std())

            

#Loop through models and run cross val score        

for model_name in [LinearRegression, DecisionTreeRegressor, linear_model.Lasso, GradientBoostingRegressor, RandomForestRegressor, SVR, XGBRegressor]:

    model = bulid_model(model_name)

    print("\n=====================================================================")

    print(model_name,"\n")

    run_cross_val_score(model, X_train, y_train, 5)    
print("The chosen models are: \n\t 1.) DecisionTreeRegressor \n\t 2.) SVR \n\t 3.) XGBRegressor \n\t 4.) RandomForestRegressor  \n\t 5.) LinearRegression")
# criterion (default='mse') - The function to measure the quality : 'mse', 'friedman_mse','mae'

#        1.) mse - mean squared error.

#        2.) friedman_mse - mean squared error : uses mean squared error with Friedman’s improvement score for potential splits.

#        3.) mse - mean squared error : mean absolute error.



# splitter (default='best') - The strategy used to choose the split at each node

#        1.) best - to choose the best split.

#        2.) random -  to choose the best random split.



# max_depth (default='None') - The maximum depth of the tree.



# min_samples_split (default=2) - The minimum number of samples required to split an internal node.



# max_features (default='None') - The number of features to consider when looking for the best split.          

#        int value = then consider max_features features at each split.

#        "auto” = then max_features=n_features.

#        “sqrt” = then max_features=sqrt(n_features).

#        “log2” = then max_features=log2(n_features)



# presort (default=False) - Whether to presort the data to speed up the finding of best splits in fitting





def get_parameters_tuning (model, X_train, y_train, X_test, y_test, cv, params_grid):

    grid = GridSearchCV(model, params_grid, cv=cv, scoring='neg_mean_squared_error')

    grid.fit(X_train, y_train)

    print('Best cross validation score: {:.2f}'.format(np.sqrt(abs(grid.best_score_))))

    print('Best parameters:', grid.best_params_)

    print('Test score:', np.sqrt(abs(grid.score(X_test, y_test))))



params_grid = {'criterion':['mse','friedman_mse','mae'], 'splitter':['best','random'], 'max_depth':[10,100,1000], 

               'min_samples_split':[5,10,20,40,80,160], 'min_samples_leaf':[1,2,3,4,5,10], 

               'max_features':['auto','sqrt','log2'], 'presort':[True,False] }



get_parameters_tuning(DecisionTreeRegressor(), X_train, y_train, X_test, y_test, 5, params_grid)
decision_tree_regressor_model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, min_samples_split=40, presort=True)

decision_tree_regressor_model.fit(X_train, y_train)
dt_predictions = decision_tree_regressor_model.predict(X_train)
def print_rmse(orig_values, predict_values):

    rmse = np.sqrt(MSE(orig_values, predict_values))

    print("RMSE = {:.2f}".format(rmse))
#DecisionTreeRegressor RMSE for train data

print('Predict RMSE with floting prediction : ')

print_rmse(y_train, dt_predictions)
#Display the Difference Between Predicted G3 and Actual G3

plt.figure(figsize=(15,7))

plt.plot(y_train-dt_predictions, '.b', markersize=4)

plt.xlabel('Data')

plt.ylabel('Actual Difference')

plt.title('Display the Difference Between Predicted G3 and Actual G3')
#DecisionTreeRegressor RMSE for test data

dt_test_predictions = decision_tree_regressor_model.predict(X_test)

print('Predict RMSE with floting prediction : ')

print_rmse(y_test, dt_test_predictions)
export_graphviz(decision_tree_regressor_model, out_file ='tree.dot')  

Image(url= "tree.jpg")
#SVR

params_grid = {'gamma': [1e-3,1e-1,1e0,1e1,1e10,1e50], 'C':[1e-3,1e-1,1e0,1e1,1e10,1e50]}

get_parameters_tuning(SVR(), X_train, y_train, X_test, y_test, 5, params_grid)
svr_model = SVR(C=10.0, gamma=0.001)

svr_model.fit(X_train, y_train)

svr_predictions = svr_model.predict(X_train)



#SVR RMSE for train data

print('Predict RMSE with floting prediction for SVR model : ')

print_rmse(y_train, svr_predictions)
plt.figure(figsize=(15,7))

plt.plot(y_train-svr_predictions, '.b', markersize=4)

plt.xlabel('Data')

plt.ylabel('Actual Difference')

plt.title('SVR - Display the Difference Between Predicted G3 and Actual G3')
#SVR RMSE for test data

svr_test_predictions = svr_model.predict(X_test)

print('Predict RMSE with floting prediction (SVR-test) : ')

print_rmse(y_test, svr_test_predictions)
#XGBRegressor

grid_params = {'max_depth':[2,3,5,7], 'learning_rate':[0.001,0.01,0.1], 'n_estimators':[50,100,200]}

get_parameters_tuning(XGBRegressor(), X_train, y_train, X_test, y_test, 5, grid_params)
model_XGBRegressor = XGBRegressor(max_depth=2,learning_rate=0.1, n_estimators=50)

model_XGBRegressor.fit(X_train,y_train)

xgbregressor_train_predictions = model_XGBRegressor.predict(X_train)



#XGBRegressor RMSE for train data

print('Predict RMSE with floting prediction for XGBRegressor model : ')

print_rmse(y_train, xgbregressor_train_predictions)
plt.figure(figsize=(15,7))

plt.plot(y_train-xgbregressor_train_predictions, '.b', markersize=4)

plt.xlabel('Data')

plt.ylabel('Actual Difference')

plt.title('XGBRegressor - Display the Difference Between Predicted G3 and Actual G3')
#XGBRegressor RMSE for test data

xgbregressor_test_predictions = model_XGBRegressor.predict(X_test)

print('Predict RMSE with floting prediction (XGBRegressor-test) : ')

print_rmse(y_test, xgbregressor_test_predictions)
#RandomForestRegressor

params_grid = {'n_estimators':[5,10,50,100,500],'max_depth':[1,3,5,7],'min_samples_leaf':[5]}

get_parameters_tuning(RandomForestRegressor(), X_train, y_train, X_test, y_test, 5, params_grid)
best_RFR = RandomForestRegressor(n_estimators=500, max_depth=7, min_samples_leaf=5)

best_RFR.fit(X_train, y_train)



randomforest_train_predictions = best_RFR.predict(X_train)

print('Predict RMSE with floting prediction (RandomForestRegressor-train) : ')

print_rmse(y_train, randomforest_train_predictions)



randomforest_test_predictions = best_RFR.predict(X_test)

print('\nPredict RMSE with floting prediction (RandomForestRegressor-test) : ')

print_rmse(y_test, randomforest_test_predictions)

params_grid = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

get_parameters_tuning(LinearRegression(), X_train, y_train, X_test, y_test, 5, params_grid)
linear_regression_model = LinearRegression()

linear_regression_model.fit(X_train,y_train)

linear_regression_predictions = linear_regression_model.predict(X_train)



print('Predict RMSE with floting prediction (LinearRegression-train) : ')

print_rmse(y_train, linear_regression_predictions)





linear_regression_test_predictions = linear_regression_model.predict(X_test)

print('\nPredict RMSE with floting prediction (LinearRegression-test) : ')

print_rmse(y_test, linear_regression_test_predictions)

#Model ensemble

model_ensemble_df = pd.DataFrame(randomforest_test_predictions,columns=['RF'])

model_ensemble_df['XGB'] = xgbregressor_test_predictions

model_ensemble_df['SVR'] = svr_test_predictions

model_ensemble_df['AVG'] = model_ensemble_df.median(axis=1)



print_rmse(y_test, model_ensemble_df['AVG'])