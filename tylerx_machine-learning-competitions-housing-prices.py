import pandas as pd



#Load data

train_orig = pd.read_csv('../input/train.csv')

test_orig = pd.read_csv('../input/test.csv')



print(train_orig.shape,test_orig.shape)
train_orig.info()

print('-'*40)

test_orig.info()
# Create label Y

Y = train_orig.SalePrice



# Clean train and test data

train=train_orig.drop(['Id','SalePrice'],axis=1)

test=test_orig.drop(['Id'],axis=1)
#encode and creat X

train_encoded=pd.get_dummies(train)

test_encoded = pd.get_dummies(test)



final_train, final_test = train_encoded.align(test_encoded, join='left', axis=1)



X=final_train
#show columns with missing data

#train_column_with_missing = (final_train.isnull().sum())

#print('final_train columns with missing data:' + str(dict(train_column_with_missing[train_column_with_missing > 0])))



#test_column_with_missing = (final_test.isnull().sum())

#print('final_test has columns with missing data'+ str(dict(test_column_with_missing[test_column_with_missing > 0])))
# Split train data into validation and training data

from sklearn.model_selection import train_test_split



train_X, val_X, train_Y, val_Y = train_test_split(X, Y,random_state=1, test_size=0.25)
#cross validation on XGBRegressor

#from xgboost import XGBRegressor

#from sklearn.pipeline import make_pipeline

#from sklearn.impute import SimpleImputer

#from sklearn.model_selection import cross_val_score



#my_pipeline_imp=make_pipeline(SimpleImputer(), XGBRegressor(n_estimators=1000, learn_rate=0.05))

#scores=cross_val_score(my_pipeline_imp, X, y, scoring='neg_mean_absolute_error', cv=5)

#print(scores)

#print('Mean Absolute Error with impu%.2f' %(-1 * scores.mean()))



#my_pipeline_noimp=make_pipeline(XGBRegressor())

#scores=cross_val_score(my_pipeline_noimp, X, y, scoring='neg_mean_absolute_error', cv=5)

#print(scores)

#print('Mean Absolute Error without imp %.2f' %(-1 * scores.mean()))
#my_pipeline_imp2=make_pipeline(SimpleImputer(), XGBRegressor(n_estimators=1000, learn_rate=0.05))

#scores=cross_val_score(my_pipeline_imp2, X, y, scoring='neg_mean_absolute_error', cv=5)

#print(scores)

#print('Mean Absolute Error with impu2 %.2f' %(-1 * scores.mean()))
#Create a measurer

from xgboost import XGBRegressor



def score_dataset(train_X, val_X, train_y, val_y):

    model = XGBRegressor(n_estimators=1000, learn_rate=0.05,random_state=29)

    model.fit(train_X, train_y)

    preds = model.predict(val_X)

    return mean_absolute_error(val_y, preds)
#drop missing values



'''from xgboost import XGBRegressor

from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score



cols_with_missing = [col for col in X.columns 

                               if X[col].isnull().any()]

X_noMis = X.drop(cols_with_missing, axis=1)

my_pipeline_drop=make_pipeline(XGBRegressor(n_estimators=1000, learn_rate=0.05))

scores=cross_val_score(my_pipeline_drop, X_noMis, y, scoring='neg_mean_absolute_error', cv=5)

print(scores)

print('Mean Absolute Error with drop %.2f' %(-1 * scores.mean()))

'''

from sklearn.metrics import mean_absolute_error

cols_with_missing = [col for col in train_X.columns 

                                 if train_X[col].isnull().any()]

train_X_noMis = train_X.drop(cols_with_missing, axis=1)

val_X_noMis  = val_X.drop(cols_with_missing, axis=1)

print("Mean Absolute Error from dropping columns with Missing Values:")

print(score_dataset(train_X_noMis, val_X_noMis, train_Y, val_Y))
#imputation

from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer

from xgboost import XGBRegressor



my_imputer = SimpleImputer()

train_X_impu = my_imputer.fit_transform(train_X)

val_X_impu = my_imputer.transform(val_X)

print("Mean Absolute Error from Imputation:")

print(score_dataset(train_X_impu,val_X_impu, train_Y, val_Y))



'''my_pipeline_impute=make_pipeline(SimpleImputer(), XGBRegressor(n_estimators=1000, learn_rate=0.05))

scores=cross_val_score(my_pipeline_impute, X, y, scoring='neg_mean_absolute_error', cv=5)

print(scores)

print('Mean Absolute Error with imputer %.2f' %(-1 * scores.mean()))'''
# impute and fit&predict using pipeline

#my_pipeline=make_pipeline(SimpleImputer(),XGBRegressor(n_estimators=1000, learn_rate=0.05) )

#my_pipeline.fit(train_X,train_y)

#preds=my_pipeline.predict(val_X)

#mean_absolute_error(val_y, preds)
#imputation plus

'''X_impuPlus = X.copy()

cols_with_missing = (cl for cl in X.columns 

                              if X[cl].isnull().any())

for c in cols_with_missing:

   X_impuPlus[c + '_was_missing'] = X_impuPlus[c].isnull()



my_pipeline_imputePlus=make_pipeline(SimpleImputer(), XGBRegressor(n_estimators=1000, learn_rate=0.05))

scores=cross_val_score(my_pipeline_imputePlus, X_impuPlus, y, scoring='neg_mean_absolute_error', cv=5)

print(scores)

print('Mean Absolute Error with imputerPlus %.2f' %(-1 * scores.mean()))'''



train_X_impuPlus = train_X.copy()

val_X_impuPlus = val_X.copy()



cols_with_missing = (col for col in train_X.columns 

                               if train_X[col].isnull().any())

for col in cols_with_missing:

   train_X_impuPlus[col + '_was_missing'] = train_X_impuPlus[col].isnull()

   val_X_impuPlus[col + '_was_missing'] = val_X_impuPlus[col].isnull()



# Imputation

my_imputer = SimpleImputer()

train_X_impuPlus = my_imputer.fit_transform(train_X_impuPlus)

val_X_impuPlus = my_imputer.transform(val_X_impuPlus)



print("Mean Absolute Error from Imputation while Track What Was Imputed:")

print(score_dataset(train_X_impuPlus, val_X_impuPlus,train_Y, val_Y))
#drop columns with missing

#cols_with_missing = [col for col in X.columns 

                             #  if X[col].isnull().any()]

#X_noMis = X.drop(cols_with_missing, axis=1)



#test_noMis=final_test.drop(cols_with_missing, axis=1)
#show columns with missing data

#train_column_with_missing = (X_noMis.isnull().sum())

#print('final_train columns with missing data:' + str(dict(train_column_with_missing[train_column_with_missing > 0])))



#test_column_with_missing = (test_noMis.isnull().sum())

#print(test_column_with_missing[test_column_with_missing > 0])
#Imputation of test and train data

X_umimputed = X.copy()

test_umimputed = final_test.copy()

#print('train_umimputed.shape'+ str(X_umimputed.shape))

#print('test_umimputed.shape'+str(test_umimputed.shape))



#train_cols_with_missing = [col for col in train_umimputed.columns 

                                 #if train_umimputed[col].isnull().any()]

#print('+train_cols_with_missing'+ str(train_cols_with_missing))

#for col in train_cols_with_missing:

   # train_umimputed[col + '_was_missing'] = train_umimputed[col].isnull()

#print('train_umimputed.shape'+ str(train_umimputed.shape))



#test_cols_with_missing = (cl for cl in test_umimputed.columns 

                                 #if test_umimputed[cl].isnull().any())

#print(list(test_cols_with_missing))

#for cl in test_cols_with_missing:

   # test_umimputed[cl + '_was_missing'] = test_umimputed[cl].isnull()

#print('test_umimputed.shape'+ str(test_umimputed.shape))



# Imputation

my_imputer = SimpleImputer()

X_imputed = my_imputer.fit_transform(X_umimputed)

test_imputed = my_imputer.transform(test_umimputed)
# Split imputed train data into validation and training data

train_X_final, val_X_final, train_Y_final, val_Y_final = train_test_split(X_imputed, Y,random_state=1)
'''from xgboost import XGBRegressor

from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score



my_pipeline_x=make_pipeline(SimpleImputer(), XGBRegressor(n_estimators=100, learn_rate=0.05))

scores=cross_val_score(my_pipeline_x, X_noMis, y, scoring='neg_mean_absolute_error', cv=5)

print(scores)

print('Mean Absolute Error x %.2f' %(-1 * scores.mean()))'''
'''from sklearn.ensemble import RandomForestRegressor

my_pipeline_R=make_pipeline(SimpleImputer(), RandomForestRegressor(n_estimators=1000, max_leaf_nodes=401,random_state=3))

scoresR=cross_val_score(my_pipeline_R, X_noMis, y, scoring='neg_mean_absolute_error', cv=5)

print(scoresR)

print('Mean Absolute Error RF %.2f' %(-1 * scoresR.mean()))'''
#from sklearn.tree import DecisionTreeRegressor



#my_pipeline_D=make_pipeline(SimpleImputer(), DecisionTreeRegressor(random_state=3))

#scoresD=cross_val_score(my_pipeline_D, X_noMis, y, scoring='neg_mean_absolute_error', cv=5)

#print(scoresD)

#print('Mean Absolute Error DT %.2f' %(-1 * scoresD.mean()))
#Random Forest Model

from sklearn.ensemble import RandomForestRegressor



rf_model = RandomForestRegressor(n_estimators=100, max_leaf_nodes=401,random_state=3)

rf_model.fit(train_X_final, train_Y_final)

rf_val_predictions = rf_model.predict(val_X_final)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_Y_final)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))



rf_model = RandomForestRegressor(n_estimators=100,max_leaf_nodes=4010,random_state=3)

rf_model.fit(train_X_final, train_Y_final)

rf_val_predictions = rf_model.predict(val_X_final)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_Y_final)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))



rf_model = RandomForestRegressor(n_estimators=100,max_leaf_nodes=20000,random_state=29)

rf_model.fit(train_X_final, train_Y_final)

rf_val_predictions = rf_model.predict(val_X_final)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_Y_final)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
#XGBoost

from xgboost import XGBRegressor

xg_model = XGBRegressor(n_estimators=1000, learn_rate=0.05, random_state=1)

# Add silent=True to avoid printing out updates with each cycle

xg_model.fit(train_X_final, train_Y_final, verbose=False, early_stopping_rounds=500, 

             eval_set=[(val_X_final, val_Y_final)])

predictions = xg_model.predict(val_X_final)

print("Mean Absolute Error : {:,.0f}".format(mean_absolute_error(predictions, val_Y_final)))
'''xg_model = XGBRegressor(n_estimators=1000, learn_rate=0.05)

xg_model.fit(train_X_final, train_y_final, verbose=False, early_stopping_rounds=1000, 

             eval_set=[(val_X_final, val_y_final)])

predictions = xg_model.predict(val_X_final)

print("Mean Absolute Error 1: {:,.0f}".format(mean_absolute_error(predictions, val_y_final)))



xg_model = XGBRegressor(n_estimators=500, learn_rate=0.01)

xg_model.fit(train_X_final, train_y_final, verbose=False, early_stopping_rounds=5, 

             eval_set=[(val_X_final, val_y_final)])

predictions = xg_model.predict(val_X_final)

print("Mean Absolute Error 2: {:,.0f}".format(mean_absolute_error(predictions, val_y_final)))



xg_model = XGBRegressor(n_estimators=100, learn_rate=0.001)

xg_model.fit(train_X_final, train_y_final, verbose=False, early_stopping_rounds=7000, 

             eval_set=[(val_X_final, val_y_final)])

predictions = xg_model.predict(val_X_final)

print("Mean Absolute Error 3: {:,.0f}".format(mean_absolute_error(predictions, val_y_final)))'''
from sklearn.model_selection import cross_val_score



my_pipeline=make_pipeline(XGBRegressor(n_estimators=1000, learn_rate=0.05, random_state=29))

scores=cross_val_score(my_pipeline, X_imputed, Y, scoring='neg_mean_absolute_error', cv=5)

print(scores)

print('Mean Absolute Error with %.2f' %(-1 * scores.mean()))
# To improve accuracy, create a new Random Forest model which you will train on all training data

from xgboost import XGBRegressor

from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score



model_full = XGBRegressor(n_estimators=1000, learn_rate=0.05, random_state=29)

model_full.fit(X_imputed, Y,verbose=False, early_stopping_rounds=500, 

             eval_set=[(val_X_final, val_Y_final)])

pred = model_full.predict(test_imputed)
#drop missing values in test data

#test_with_missing = [col for col in predictors.columns 

                                 #if predictors[col].isnull().any()]

#predictors_noMis = predictors.drop(test_with_missing, axis=1)
#missing_val_count_by_column = (predictors_noMis.isnull().sum())

#print(missing_val_count_by_column[missing_val_count_by_column > 0])
# make predictions which we will submit. 

#test_preds = xg_model_full_data.predict(test_imputed)



# The lines below shows you how to save your data in the format needed to score it in the competition

output = pd.DataFrame({'Id': test_orig.Id,

                       'SalePrice': pred})



output.to_csv('submission10.csv', index=False)