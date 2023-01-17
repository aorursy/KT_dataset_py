
# Regular EDA and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# preprocessor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Models from Scikit-Learn
from sklearn.ensemble import RandomForestRegressor

# Model Evaluations
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error,mean_squared_log_error,mean_absolute_error,make_scorer
#Pipeline
from sklearn.pipeline import Pipeline
plt.style.use('seaborn-whitegrid')
from datetime import datetime
# combined dataset of training and validation set
df = pd.read_csv("../input/bluebook-for-bulldozers/TrainAndValid.csv",parse_dates=['saledate'],low_memory=False) 
# test set
test_df = pd.read_csv("../input//bluebook-for-bulldozers/Test.csv",parse_dates=['saledate'],low_memory=False)
# sorting df according to the saledate
df.sort_values(by='saledate',inplace=True)
df.head().T 
df.info() # most of the features are having object DataType
test_df.info()
# shape of the dataframe
df.shape
test_df.shape
df.isna().sum()
test_df.isna().sum()
# visualizing missing entries
df_missing_percentage = ((df.isna().sum()/df.shape[0])*100)
test_df_missing_percentage = ((df.isna().sum()/df.shape[0])*100)
pd.DataFrame(df_missing_percentage,columns=['missing%']).sort_values(by='missing%').plot(kind='barh',figsize=(7,15));
plt.xticks(fontsize = 15);
plt.yticks(fontsize = 10);
pd.DataFrame(test_df_missing_percentage,columns=['missing%']).sort_values(by='missing%').plot(kind='barh',figsize=(7,15));
plt.xticks(fontsize = 15);
plt.yticks(fontsize = 10);
# Concatinatng all data 
# test_df has no SalePrice column , so its data points will have NaN in its SalePrice column when cancatenated with df 
Concat = pd.concat((df,test_df),axis = 0).reset_index(drop=True)

# Converting all columns with object dtype to category dtype
for label,content in Concat.items() :
    if pd.api.types.is_object_dtype(content):
        Concat[label] = content.astype('category')
        
# Enriching features
Concat['year'] = Concat.saledate.dt.year
Concat['month']= Concat.saledate.dt.month
Concat['day']= Concat.saledate.dt.day
cat=[] # list for storing all columns with 'cstegory' dtype
cat_missing = [] # list for storing columns with 'category' dtype and having missing values
num_missing = [] # list for storing columns with 'numerical' dtype and having missing values
for label,content in Concat.items():
    
    if pd.api.types.is_numeric_dtype(content): # checking for numerical features
        if content.isna().sum() > 0: # checking if the feature has any missing values
            Concat[f'{label}_ismissing'] = content.isna()
            num_missing.append(label)
            
    if pd.api.types.is_categorical_dtype(content): # checking for categorical features
        cat.append(label) 
        if content.isna().sum() > 0: # checking if the feature has any missing values
            Concat[f'{label}_ismissing'] = content.isna()
            cat_missing.append(label)
            
cat_not_missing = list(set(cat) - set(cat_missing))
# For missing values in categorical datatype, by default `-1` is assigned for its code, so adding 1 before creating new column
Concat[cat_missing] = Concat[cat_missing].apply(lambda i : i.cat.codes+1)

# For features with no missing values, simply assigning code
Concat[cat_not_missing] = Concat[cat_not_missing].apply(lambda i : i.cat.codes)
(Concat.isna().sum() !=0 ).sum() # out which one is SalePrice , which will not be considered
train_df = Concat.loc[Concat.saledate.dt.year < 2012, :].drop('saledate', axis=1)

valid_df = Concat.loc[Concat.saledate <= pd.Timestamp(
    year=2012, month=4, day=30)].loc[Concat.saledate >= pd.Timestamp(year=2012, month=1, day=1)].drop('saledate', axis=1)

test_df = Concat.loc[Concat.saledate >=
                     pd.Timestamp(year=2012, month=4, day=30), :].drop(['SalePrice','saledate'], axis=1)
train_df.shape
test_df.shape
valid_df.shape
train_df[num_missing].isna().sum()
valid_df[num_missing].isna().sum()
num_imputer = SimpleImputer(strategy='median')
transformer = ColumnTransformer(transformers=[('num_missing',num_imputer,train_df.columns)],remainder='passthrough',)

train_df_filled = transformer.fit_transform(train_df) # fitting on training data 
valid_df_filled = transformer.transform(valid_df) # transforming test based on training data to avoid data leakage

train_df_filled = pd.DataFrame(train_df_filled,columns=train_df.columns)
valid_df_filled = pd.DataFrame(valid_df_filled,columns=valid_df.columns)
train_df_filled
train_df_filled[num_missing].isna().sum()
valid_df_filled[num_missing].isna().sum()
# separating features and labels
X_train_filled,y_train_filled = train_df_filled.drop(['SalePrice'],axis=1),train_df_filled.SalePrice 
X_valid_filled,y_valid_filled = valid_df_filled.drop(['SalePrice'],axis=1),valid_df_filled.SalePrice

X_train,y_train = train_df.drop(['SalePrice'],axis=1),train_df_filled.SalePrice
X_valid,y_valid = valid_df.drop(['SalePrice'],axis=1),valid_df_filled.SalePrice
# array to store diffrent hyperparameters of RandomForestClassifier and respective scores
models=[]
# function to evaluate diffrent metrics 
def evaluate(model, X_valid, y_valid):
    y_pred = model.predict(X_valid)
    scores = {
        'R2': model.score(X_valid, y_valid),
        'MAE': mean_absolute_error(y_valid, y_pred),
        'RMLE': np.sqrt(mean_squared_log_error(y_valid, y_pred))}
    return(scores)
# function to compile hyperparameters and scores of a model
def store_result(model,params, X,y):
    y_pred = model.predict(X)
    scores = {
        'R2': model.score(X, y),
        'MAE': mean_absolute_error(y, y_pred),
        'RMLE': np.sqrt(mean_squared_log_error(y, y_pred))}
    model = {
        'scores': scores,
        'params': params}
    if model not in models:
        models.append(model)
    return(model)
rfr = RandomForestRegressor(n_jobs=-1,n_estimators=100,max_depth=4)
rfr.fit(X_train_filled,y_train_filled) # fitting fileed training data
evaluate(rfr,X_valid_filled,y_valid_filled)
param_grid={
    'randomforestregressor__n_estimators':np.arange(90,150,10),
    'randomforestregressor__max_depth':[None],
    'randomforestregressor__min_samples_split':np.arange(2,20,1),
    'randomforestregressor__min_samples_leaf':np.arange(1,15,1)
} 
num_imputer = SimpleImputer(strategy='median')
transformer = ColumnTransformer(transformers=[('num_missing',num_imputer,X_train.columns)])
rfr = RandomForestRegressor(random_state=23,max_samples=10000,n_jobs=-1) # tuning using only 10000 samples
rfr_pipeline= make_pipeline(transformer,rfr)
scorer_func= lambda y_true,y_pred: np.sqrt(mean_squared_log_error(y_true,y_pred)) # custom scorer function of rmsle
scorer = make_scorer(scorer_func,greater_is_better=False)
rfr_random = RandomizedSearchCV(estimator=rfr_pipeline,
                                param_distributions=param_grid,
                                cv=5,
                                n_jobs=-1,
                                n_iter=2,
                                scoring=scorer)
%%time
rfr_random.fit(X_train,y_train) #
 #fitting entire training data to the best estimator
rfr = rfr_random.best_estimator_.set_params(randomforestregressor__max_samples = None);
rfr.fit(X_train,y_train)
store_result(rfr,rfr_random.best_params_,X_valid,y_valid)
transformer = ColumnTransformer(transformers=[('num_missing',num_imputer,X_train.columns)])
rfr = RandomForestRegressor(random_state=23,max_samples=4000,n_jobs=-1)
rfr_pipeline= make_pipeline(transformer,rfr)
param_grid={
    'randomforestregressor__n_estimators':np.arange(110,121,1),
    'randomforestregressor__max_depth':[None],
    'randomforestregressor__min_samples_split':[19],
    'randomforestregressor__min_samples_leaf':[7]
}
rfr_grid = GridSearchCV(estimator=rfr_pipeline,
                                param_grid=param_grid,
                                cv=5,
                                n_jobs=-1,
                                scoring=scorer)
rfr_grid.fit(X_train,y_train)
rfr = rfr_grid.best_estimator_.set_params(randomforestregressor__max_samples = None);
rfr.fit(X_train,y_train)
store_result(rfr,rfr_grid.best_params_,X_valid,y_valid)
models
{'scores': {'R2': 0.8730819859878117,
   'MAE': 6050.945892315819,
   'RMLE': 0.24864908475502068},
  'params': {'randomforestregressor__n_estimators': 140,
   'randomforestregressor__min_samples_split': 9,
   'randomforestregressor__min_samples_leaf': 6,
   'randomforestregressor__max_depth': None}} # best found results with parameters
