%matplotlib inline

df = pd.read_csv("train.csv")
pd.options.display.max_columns = None
df.head()
df.shape
df.info()
df.isnull().sum()
df.drop(["PoolQC","Fence","MiscFeature","Alley"],axis=1,inplace=True)
df.shape
sns.heatmap(df.isnull(),yticklabels=False,cmap="YlGnBu",cbar=False)
df.head()
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].mean())
df["MasVnrType"].value_counts()
df["MasVnrType"] = df["MasVnrType"].fillna(df["MasVnrType"].mode()[0])
df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].mean())
df["FireplaceQu"].value_counts()
data_corr = df.corr()
data_corr["SalePrice"].sort_values(ascending=False)
df.drop("FireplaceQu",axis=1,inplace=True)
df["GarageType"].value_counts()
df["GarageType"] = df["GarageType"].fillna(df["GarageType"].mode()[0])
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].mean())
def x(columns_int):
    for col in columns_list:
        df["col"] = df["col"].fillna(df["col"].mean())
    return df
df["GarageFinish"].value_counts()
df["GarageFinish"] = df["GarageFinish"].fillna(df["GarageFinish"].mode()[0])
df["GarageQual"] = df["GarageQual"].fillna(df["GarageQual"].mode()[0])
df["GarageCond"] = df["GarageCond"].fillna(df["GarageCond"].mode()[0])
sns.heatmap(df.isnull(),yticklabels=False,cmap="YlGnBu",cbar=False)
df.info(verbose=None)
df["BsmtQual"] = df["BsmtQual"].fillna(df["BsmtQual"].mode()[0])
df["BsmtCond"] = df["BsmtCond"].fillna(df["BsmtCond"].mode()[0])
df["BsmtExposure"] = df["BsmtExposure"].fillna(df["BsmtExposure"].mode()[0])
df["BsmtFinType1"] = df["BsmtFinType1"].fillna(df["BsmtFinType1"].mode()[0])
df["BsmtFinType2"] = df["BsmtFinType2"].fillna(df["BsmtFinType2"].mode()[0])
df.info(verbose=None)
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
df_objects = df.select_dtypes(include=['object']).copy()
df_objects.shape
df_objects.columns
columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition']
df.dropna(inplace=True)
df.shape
len(columns)
test_df = pd.read_csv("finaltest.csv")
test_df.head()
final_df = pd.concat([df,test_df],axis=0)
final_df.shape
def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final
final_df_hot=category_onehot_multcols(columns)
final_df_hot.shape
final_df_hot =final_df_hot.loc[:,~final_df_hot.columns.duplicated()]
final_df_hot.shape
df_Train=final_df_hot.iloc[:1460,:]
df_Test=final_df_hot.iloc[1460:,:]
df_Test.drop(['SalePrice'],axis=1,inplace=True)
X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
import xgboost
clf=xgboost.XGBRegressor()
clf.fit(X_train,y_train)
import pickle
filename = 'finalized_model.pkl'
pickle.dump(clf, open(filename, 'wb'))
y_pred = clf.predict(df_Test)
y_pred
pred = pd.DataFrame(y_pred)
sub_df=pd.read_csv("sample_submission.csv")
datasets = pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns = ['Id','SalePrice']
datasets.to_csv("sample_submission1.csv",index=False)
import xgboost
regressor=xgboost.XGBRegressor()
booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]

n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }
from sklearn.model_selection import RandomizedSearchCV
# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=3, n_iter=20,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)
random_cv.fit(X_train,y_train)
random_cv.best_estimator_
clf_new_xg=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.1, max_delta_step=0, max_depth=3,
             min_child_weight=3, monotone_constraints=None,
             n_estimators=100, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
clf_new_xg.fit(X_train,y_train)
import pickle
filename = 'finalized_model_xg_para_opt.pkl'
pickle.dump(clf, open(filename, 'wb'))
y_pred_new = clf_new_xg.predict(df_Test)
y_pred_new
pred_new = pd.DataFrame(y_pred_new)
sub_df=pd.read_csv("sample_submission.csv")
datasets = pd.concat([sub_df['Id'],pred_new],axis=1)
datasets.columns = ['Id','SalePrice']
datasets.to_csv("sample_submission_xgb_para_opt.csv",index=False)
X_train.shape,y_train.shape
from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

model = Sequential()
model.add(Dense(output_dim=50,init = 'he_uniform',activation='relu',input_dim = 177))
model.add(Dense(output_dim=25,init = 'he_uniform',activation='relu'))
model.add(Dense(output_dim=50,init = 'he_uniform',activation='relu'))
model.add(Dense(output_dim = 1, init = 'he_uniform'))
model.compile(loss= root_mean_squared_error,optimizer='Adamax')

history = model.fit(X_train.values, y_train.values,validation_split=0.20, batch_size = 10, nb_epoch = 1000)
ann_pred=model.predict(df_Test)
ann_pred
pred_new_ann = pd.DataFrame(ann_pred)
sub_df=pd.read_csv("sample_submission.csv")
datasets = pd.concat([sub_df['Id'],pred_new_ann],axis=1)
datasets.columns = ['Id','SalePrice']
datasets.to_csv("sample_submission_ann.csv",index=False)
