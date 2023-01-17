import numpy as np, pandas as pd

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_log_error  ####USE param squared=False to evaluate using RMSE



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



##### TEST

test_dataset=pd.read_csv(os.path.join(dirname,'test.csv'),index_col='Id')
def read_datasets():

    ''' READ THE DATA FROM train.csv FILE AND RETURN DATASETS '''

    train_dataset=pd.read_csv(os.path.join(dirname,'train.csv'),index_col='Id')

    fulltrain_y=train_dataset.SalePrice

    fulltrain_X=train_dataset.drop(columns=['SalePrice'])

    

    #''' DROP COLUMNS WITH 200+ N/A VALUES AFTER MEANINGFULLY VERIFYING DESCRIPTION AND RELEVANCE TO PRICE PREDICTION '''

    x=train_dataset.isna().sum()

    verysparse_columns=list(x[x>200].index)

    print("Columns with 200+ N/A values= ",verysparse_columns)

    #print("I think we can drop off 'PoolQC' and 'Alley'.")

    fulltrain_X=fulltrain_X[set(fulltrain_X.columns)-set(['PoolQC','Alley','FireplaceQu','Fence'])]

    

    #''' REPLACE N/A IN REST CATEGORICAL VARIABLES WITH "Other" VALUE '''

    #rest_categorical_sparse_columns=['MiscFeature','Fence','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','BsmtExposure','BsmtFinType2','BsmtQual','BsmtCond','BsmtFinType1','MasVnrType','Electrical']

    #for col in rest_categorical_sparse_columns:

    #    fulltrain_X[col].fillna(value='Other',inplace=True)

    

    train_X,val_X, train_y,val_y=train_test_split(fulltrain_X,fulltrain_y,random_state=0)

    return (train_X,train_y,val_X,val_y)



train_X,train_y,val_X,val_y=read_datasets()





''' EXTRACT THE NUMERICAL FIELDS AND CATEGORICAL FIELDS WITH LOW CARDINALITY '''

numerical_cols=list(train_X.select_dtypes(exclude=['object']).columns)

categorical_cols=list(train_X.select_dtypes(include=['object']).columns)

lowcardinality_categorical_cols = [col for col in categorical_cols if train_X[col].nunique()<10]



required_cols=numerical_cols+lowcardinality_categorical_cols

train_X=train_X[required_cols]

val_X=val_X[required_cols]



print("\nTRAIN_X:")

train_X.describe()
def exec_pipeline(xgb_estimators, xgb_max_depth, xgb_learning_rate):

    extremeGradientBoost_RegressorModel=XGBRegressor(n_estimators=xgb_estimators,max_depth=xgb_max_depth,learning_rate=xgb_learning_rate,random_state=0)



    numerical_transformer=SimpleImputer(strategy='mean')

    categorical_transformer=Pipeline(steps=[

        #Commenting out this line since we handled NULL values in first function itself

        ('impute_categ_vals',SimpleImputer(strategy='most_frequent')),

        ('onehotenc',OneHotEncoder(sparse=False,handle_unknown='ignore'))

    ])



    preprocessor=ColumnTransformer(transformers=[

        ('numerical_preprocess', numerical_transformer, numerical_cols),

        ('categorical_preprocess', categorical_transformer, lowcardinality_categorical_cols)

    ])



    model_pipeline=Pipeline(steps=[

        ('preprocess_fields',preprocessor),

        ('xgbregr_model',extremeGradientBoost_RegressorModel)

    ])



    model_pipeline.fit(train_X,train_y)

    predictions=model_pipeline.predict(val_X)

    

    rmse=np.sqrt(mean_squared_log_error(predictions,val_y))

    print('N_ESTIMATORS={} , MAX_DEPTH={} , LEARNING_RATE={} ==> RMSE={}'.format(xgb_estimators, xgb_max_depth, xgb_learning_rate,rmse))

    return rmse
'''losses=[]

#1500: 0.123142

#1600: 0.1218009

estimators=[1200,1400,1600,1800,2000,2100,2200,2300,2400,2500]



#0.0084:   0.124657

#0.008372: 0.1218009

#learning_rates=[0.008372, 0.0083718, 0.0083722, 0.0083716, 0.00837224, 0.0083714, 0.0083728]

learning_rates=[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.0089,0.009]



#6: 0.1234141

max_depths=[6,8,10,12,14,16,18,20]

for lr in learning_rates:

    for e in estimators:

        for d in max_depths:

            losses.append((e,d,lr,exec_pipeline(e,d,lr)))

        print("")

    print("-")



print(losses)

exp_losses=pd.DataFrame(losses,columns=['N_ESTIMATORS','MAX_DEPTH','LEARNING_RATE','RMSE'])

exp_losses.to_csv('exp_losses.csv')

'''
def exec_pipeline_with_kcross_val(xgb_estimators, xgb_max_depth, xgb_learning_rate,kfolds=4):

    extremeGradientBoost_RegressorModel = XGBRegressor(n_estimators = xgb_estimators, max_depth = xgb_max_depth, learning_rate = xgb_learning_rate,

                                                       random_state=0)



    numerical_transformer=SimpleImputer(strategy='mean')

    categorical_transformer=Pipeline(steps=[

        ('impute_categ_vals',SimpleImputer(strategy='most_frequent')),

        ('onehotenc',OneHotEncoder(sparse=False,handle_unknown='ignore'))

    ])



    preprocessor=ColumnTransformer(transformers=[

        ('numerical_preprocess', numerical_transformer, numerical_cols),

        ('categorical_preprocess', categorical_transformer, lowcardinality_categorical_cols)

    ])



    model_pipeline=Pipeline(steps=[

        ('preprocess_fields',preprocessor),

        ('xgbregr_model',extremeGradientBoost_RegressorModel)

    ])



    kf=KFold(shuffle=True,n_splits=kfolds,random_state=0)

    fullX=pd.concat([train_X,val_X],axis=0)

    fully=pd.concat([train_y,val_y],axis=0)

    model_pipeline.fit(train_X,train_y)

    rmse=np.sqrt(-1*cross_val_score(model_pipeline, fullX, fully, cv=kf, scoring='neg_mean_squared_log_error'))

    

    print('N_ESTIMATORS={} , MAX_DEPTH={} , LEARNING_RATE={} ==> RMSE={}'.format(xgb_estimators, xgb_max_depth, xgb_learning_rate,rmse))

    return (np.mean(rmse),model_pipeline)

'''estimators=[1200,1400,1600,1800,2000,2100,2200,2300,2400,2500]

kfold_losses=[]

for e in estimators:

    kfold_losses.append([e,14,0.0089,4,(exec_pipeline_with_kcross_val(e,14,0.0089,4))])'''

#0.1381 => 0.133829

rmse,model_pipeline=exec_pipeline_with_kcross_val(1400,6,0.0081,4)

print(rmse)
test_dataset=test_dataset[set(test_dataset.columns)-set(['PoolQC','Alley','FireplaceQu','Fence'])]

predictions=model_pipeline.predict(test_dataset)

predictions

submission_df=pd.DataFrame()

submission_df['Id']=test_dataset.index

submission_df['SalePrice']=predictions



submission_df.to_csv('submission.csv',index=False)
for i in submission_df.SalePrice:

    print(i)