# Essential
import numpy as np 
import pandas as pd 

# Plotting
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Models
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from mlxtend.regressor import StackingCVRegressor

# Cross Validation & Metrics
from sklearn.model_selection import GridSearchCV,cross_val_score,KFold
from sklearn.metrics import mean_squared_error

# Preprocessing
from sklearn.preprocessing import StandardScaler,RobustScaler

# Misc
from sklearn.pipeline import make_pipeline
from collections import Counter
# Read train,test and sample files and set custom seaborn graph style
train = pd.read_csv('../input/boston20200827/Boston_train.csv')
test = pd.read_csv('../input/boston20200827/Boston_test.csv')
sample = pd.read_csv('../input/boston20200827/Boston_submission.csv')

sns.set_style('darkgrid')
# Read the first 5 rows of the train dataset
train.head()
# Get some basic information about the dataset's features
train.info()
# Get some basic statistical information about the features
train.describe()
# See the correlation between the target and the other features, ordered by positive correlation
train.drop('ID',axis=1).corr()['PRICE'].sort_values(ascending=False)[1:]
# See the amount of unique values
train.nunique()
# Display a boxplot for every feature in the dataset
for feature in list(train.columns[1:]):
    sns.boxplot(train[feature])
    plt.show()
# Display heatmap of pearson correlation of all the features
plt.figure(figsize=(20,10))
sns.heatmap(train.corr(),annot=True,cmap='coolwarm')
plt.show()
# Display a linear model plot of the LSAT feature against the PRICE feature
sns.lmplot('LSTAT','PRICE',data=train,line_kws={'color':'black'},scatter_kws={'color':'red'})
plt.show()
# Seaborn jointplot of LSAT against plot in hex format
sns.jointplot('LSTAT','PRICE',data=train,kind='hex')
plt.show()
# Distribution plot of target variable
sns.distplot(train['PRICE'])
plt.show()
# Detect outliers using the Tuckey Method
def tuckey_method_outlier_detection(df,n,features):
    
    outlier_indices = []
    
    for col in features:
        Q1 = np.percentile(df[col],25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5 * IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k,v in outlier_indices.items() if v > n)
    return multiple_outliers
# Call our tuckey method function and locate outliers
outlier_indices = tuckey_method_outlier_detection(train,2,list(train.drop(['ID','PRICE'],axis=1).columns))
train.loc[outlier_indices]
# Remove the outliers
train = train.drop(outlier_indices,axis=0)
# Define our train and test set
X = train.drop(['ID','PRICE'],axis=1)
y = train['PRICE']
X_test = test.drop('ID',axis=1)
# Initalize a 5 Fold cross validation
kf = KFold(n_splits=5,shuffle=True,random_state=42)
cv_scores = []
rmse_scores = []
# Function to evaluate model using RMSE on training data and cross_val_score
def evaluate_model(model):
    model.fit(X,y)
    cv_score = np.sqrt(-cross_val_score(model,X,y,cv=kf,scoring='neg_mean_squared_error').mean())
    rmse_score = np.sqrt(mean_squared_error(y,model.predict(X)))
    
    print('RMSE on training set: ' + str(rmse_score))
    print('Cross Validaion Score: ' + str(cv_score))
    
    rmse_scores.append(rmse_score)
    cv_scores.append(cv_score)
    
# Submit desired model as a competition submission
def submit(model,model_name):
    model.fit(X,y)
    
    sample['PRICE'] = model.predict(X_test)
    sample.to_csv('submission_'+str(model_name)+'.csv',index=False)
# Define and evaluate our linear regression model
lr = make_pipeline(StandardScaler(),LinearRegression())
evaluate_model(lr)
# Define our multi layer perceptron regressor model
mlp = make_pipeline(StandardScaler(),MLPRegressor(random_state=42,solver='sgd',max_iter=4000,hidden_layer_sizes=(200,2)))
evaluate_model(mlp)
# Stack two models together
stacked_regressor = StackingCVRegressor(regressors=[
    mlp,lr
],meta_regressor=lr)

evaluate_model(stacked_regressor)
