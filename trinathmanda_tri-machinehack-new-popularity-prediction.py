!pip install pycaret

#import sys
#!{sys.executable} -m pip install --upgrade llvmlite --ignore-installed
#!{sys.executable} -m pip install --upgrade numba --ignore-installed
#!{sys.executable} -m pip install --upgrade scipy --ignore-installed
#!{sys.executable} -m pip install pycaret
#!{sys.executable} -m pip install autoplotter
#!{sys.executable} -m  pip install autoviz


# check version
from pycaret.utils import version
version()
import pandas as pd
train_data = pd.read_csv('/kaggle/input/news-popularity-prediction-machinehack/Train.csv')
test_data=pd.read_csv('/kaggle/input/news-popularity-prediction-machinehack/Test.csv')

print(train_data.shape)
print(test_data.shape)

test_data['shares'] = train_data['shares'].mean()
df = pd.concat([train_data, test_data]).reset_index(drop=True)
df.describe()

#df_report = ProfileReport(df)
#df_report.to_file(output_file='df_report.html')

df_k = df.drop(columns=['kw_max_max','kw_max_min','n_non_stop_words','n_unique_tokens'], axis=1, inplace=False)
boolean_columns=['data_channel_is_bus','data_channel_is_entertainment','data_channel_is_lifestyle','data_channel_is_socmed','data_channel_is_tech','data_channel_is_world','is_weekend','weekday_is_friday','weekday_is_monday','weekday_is_saturday','weekday_is_sunday','weekday_is_thursday','weekday_is_tuesday','weekday_is_wednesday']
for col in boolean_columns:
    df_k[col] = df_k[col].astype(int)
df1 = df_k.copy()

Q1 = df1.quantile(0.05)
Q3 = df1.quantile(0.95)
IQR = Q3 - Q1
#print(IQR)
df_without_outliers = df1[~((df1 < (Q1 - 1 * IQR)) |(df1 > (Q3 + 1 * IQR))).any(axis=1)]
df_without_outliers.describe()
df_final = df_without_outliers.round(3)
df_final.reset_index(drop=True, inplace=True)
from pycaret.regression import *
reg1 = setup(df_final, target = 'shares', sampling=False, silent=True, train_size=0.8 )
#compare_models(blacklist=['ransac', 'rf', 'et'])
svm1 =create_model('svm')
#tuned_svm = tune_model(svm1, fold=10,n_iter=10, optimize='MAE', choose_better=True )
#tuned_svm
svm_final = finalize_model(svm1)
predict_data = pd.read_csv('/kaggle/input/news-popularity-prediction-machinehack/Test.csv')
predict_data.drop(columns=['kw_max_max','kw_max_min','n_non_stop_words','n_unique_tokens'], axis=1, inplace=True)

boolean_columns=['data_channel_is_bus','data_channel_is_entertainment','data_channel_is_lifestyle','data_channel_is_socmed','data_channel_is_tech','data_channel_is_world','is_weekend','weekday_is_friday','weekday_is_monday','weekday_is_saturday','weekday_is_sunday','weekday_is_thursday','weekday_is_tuesday','weekday_is_wednesday']
for col in boolean_columns:
    predict_data[col] = predict_data[col].astype(int)
    

predict_data = predict_data.round(3)

predict_data = predict_data.reset_index(drop=True)


predictions = predict_model(svm_final, data=predict_data) 
#predictions.rename(columns = {'Label':'shares'}, inplace = True) 
a=pd.DataFrame()
a['shares']=predictions['Label']
a.to_excel('sample_submission.xlsx', index=False)