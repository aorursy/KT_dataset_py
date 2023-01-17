import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import time



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 500)

pd.set_option('display.max_rows', 1000)



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline 
bill_of_materials_df = pd.read_csv('../input/bill_of_materials.csv')

comp_adaptor_df = pd.read_csv('../input/comp_adaptor.csv')

comp_boss_df = pd.read_csv('../input/comp_boss.csv')

comp_elbow_df = pd.read_csv('../input/comp_elbow.csv')

comp_float_df = pd.read_csv('../input/comp_float.csv')

comp_hfl_df = pd.read_csv('../input/comp_hfl.csv')

comp_nut_df = pd.read_csv('../input/comp_nut.csv')

comp_other_df = pd.read_csv('../input/comp_other.csv')

comp_sleeve_df = pd.read_csv('../input/comp_sleeve.csv')

comp_straight_df = pd.read_csv('../input/comp_straight.csv')

comp_tee_df = pd.read_csv('../input/comp_tee.csv')

comp_threaded_df = pd.read_csv('../input/comp_threaded.csv')

components_df = pd.read_csv('../input/components.csv')

specs_df = pd.read_csv('../input/specs.csv')

test_set_df = pd.read_csv('../input/test_set.csv')

train_set_df = pd.read_csv('../input/train_set.csv')

tube_end_form_df = pd.read_csv('../input/tube_end_form.csv')

tube_df = pd.read_csv('../input/tube.csv')

type_component_df = pd.read_csv('../input/type_component.csv')

type_connection_df = pd.read_csv('../input/type_connection.csv')

type_end_form_df = pd.read_csv('../input/type_end_form.csv')
train_set_df.head()
train_set_df.shape
test_set_df.head()
train_label = train_set_df["cost"]



train_set_df.drop("cost", axis=1, inplace=True)

test_set_df.drop("id", axis=1, inplace=True)
#merge1: train + tube_df



merge1 = pd.merge(train_set_df, tube_df, on="tube_assembly_id")

test_merge1 = pd.merge(test_set_df, tube_df, on="tube_assembly_id")
#merge2: train + tube_df + bill_of_materials_df(bill_of_materials_summary_df)



#The 1,3,5,7...15 columns of bill_comp_types_df are informations about component_id.



#We calculate each tube_assembly uses how many different component for assembly, shown as component_series

bill_comp_types_df = bill_of_materials_df.iloc[:,[1,3,5,7,9,11,13,15]]

bill_comp_types_logical_df = ~bill_comp_types_df.isnull()

component_series = bill_comp_types_logical_df.sum(axis = 1)





#The 2,4,6,8...16 columns of bill_comp_types_df are informations about how many number of 

#components needed for assembly



#Then we calculate the total number of components needed for assembly, shown as quants_series.

bill_comp_quants_df = bill_of_materials_df.iloc[:,[2,4,6,8,10,12,14,16]]

quants_series = bill_comp_quants_df.sum(axis = 1)



bill_of_materials_summary_df = bill_of_materials_df.copy()

bill_of_materials_summary_df['type_totals'] = component_series

bill_of_materials_summary_df['component_totals'] = quants_series

bill_of_materials_summary_df['component_average_quality'] = bill_of_materials_summary_df["component_totals"] / bill_of_materials_summary_df["type_totals"]



merge2 = pd.merge(merge1, bill_of_materials_summary_df, on="tube_assembly_id")

test_merge2 = pd.merge(test_merge1, bill_of_materials_summary_df, on="tube_assembly_id")
#merge3: train + tube_df + bill_of_materials_df(bill_of_materials_summary_df) + specs_df(totals_spec)

specs_only_df = specs_df.iloc[:, 1:11]

specs_logical_df = ~specs_only_df.isnull()

specs_totals = specs_logical_df.sum(axis=1)



specs_with_totals_df = specs_df.copy()

specs_with_totals_df['spec_totals'] = specs_totals



merge3 = pd.merge(merge2, specs_with_totals_df[['tube_assembly_id', 'spec_totals']], on="tube_assembly_id")

test_merge3 = pd.merge(test_merge2, specs_with_totals_df[['tube_assembly_id', 'spec_totals']], on="tube_assembly_id")
merge3.head()
#tube_end_form_df.columns = ["end_a", "end_x_forming"]

#merge4 = pd.merge(merge3, tube_end_form_df, on="end_a")

#test_merge4 = pd.merge(test_merge3, tube_end_form_df, on="end_x")
#tube_end_form_df.columns = ["end_x", "end_x_forming"]

#merge5 = pd.merge(merge4, tube_end_form_df, on="end_x")

#test_merge5 = pd.merge(test_merge4, tube_end_form_df, on="end_x")
result = merge3.copy()

test_result = test_merge3.copy()
result.shape
result.head()
#create new date features







#create new date features



result["quote_date"] = pd.to_datetime(result["quote_date"])



result["year"] = result["quote_date"].dt.year

result["month"] = result["quote_date"].dt.month

#result["day"] = result["quote_date"].dt.day

#result["dayofweek"] = result["quote_date"].dt.dayofweek



#test data

test_result["quote_date"] = pd.to_datetime(test_result["quote_date"])



test_result["year"] = test_result["quote_date"].dt.year

test_result["month"] = test_result["quote_date"].dt.month



#code end
#create new numeric features follow its relationship 



result['bend_radius_div_wall'] = result["bend_radius"] / result["wall"]

result['diameter_div_wall'] = result["diameter"] / result["wall"]



#test data

test_result['bend_radius_div_wall'] = test_result["bend_radius"] / test_result["wall"]

test_result['diameter_div_wall'] = test_result["diameter"] / test_result["wall"]







#code end
result["same_end_form"] = (result["end_a"] == result["end_x"])



test_result["same_end_form"] = (test_result["end_a"] == test_result["end_x"])



#code end




def catch_num_tube_assembly_id(row):

    return int(row["tube_assembly_id"][-5:])



def catch_num_supplier(row):

    return int(row["supplier"][-4:])



def catch_num_material_id(row):

    if type(row["material_id"]) == float:

        return row["material_id"]

    

    else:

        return int(row["material_id"][-4:])



#create new numeric features based some categorical features



result['num_tube_assembly_id'] = result.apply (lambda row: catch_num_tube_assembly_id (row),axis=1)

result['num_supplier'] = result.apply (lambda row: catch_num_supplier (row),axis=1)

result['num_material_id'] = result.apply (lambda row: catch_num_material_id (row),axis=1)



#test data

test_result['num_tube_assembly_id'] = test_result.apply (lambda row: catch_num_tube_assembly_id (row),axis=1)

test_result['num_supplier'] = test_result.apply (lambda row: catch_num_supplier (row),axis=1)

test_result['num_material_id'] = test_result.apply (lambda row: catch_num_material_id (row),axis=1)
df1 = pd.merge(components_df, comp_adaptor_df[["component_id", "weight"]], on="component_id")

df2 = pd.merge(components_df, comp_boss_df[["component_id", "weight"]], on="component_id")

df3 = pd.merge(components_df, comp_elbow_df[["component_id", "weight"]], on="component_id")

df4 = pd.merge(components_df, comp_float_df[["component_id", "weight"]], on="component_id")

df5 = pd.merge(components_df, comp_hfl_df[["component_id", "weight"]], on="component_id")

df6 = pd.merge(components_df, comp_nut_df[["component_id", "weight"]], on="component_id")

df7 = pd.merge(components_df, comp_other_df[["component_id", "weight"]], on="component_id")

df8 = pd.merge(components_df, comp_sleeve_df[["component_id", "weight"]], on="component_id")

df9 = pd.merge(components_df, comp_straight_df[["component_id", "weight"]], on="component_id")

df10 = pd.merge(components_df, comp_tee_df[["component_id", "weight"]], on="component_id")

df11 = pd.merge(components_df, comp_threaded_df[["component_id", "weight"]], on="component_id")
frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11]



component_weight = pd.concat(frames).sort_values("component_id", ascending=True).reset_index(drop=True)[["component_id", "weight"]]
component_weight.head()
#use component_weight and bill_of_materials_df to calculate the weight of each tube, as tube_weight

















#code end
#tube_weight.head()
#add weight feature of each tube

#merge tube_weight with master dataframe 



#end code



#test data

#merge tube_weight with master dataframe 



#end code
result.head()
#drop useless features



#data = result.select_dtypes(include=['int', 'float'])

result.drop(["tube_assembly_id", "quote_date"], axis=1, inplace=True)



test_result.drop(["tube_assembly_id", "quote_date"], axis=1, inplace=True)
result.head()
train_set_df = result.copy()



test_set_df = test_result.copy()
train_set_df.head()
test_set_df.head()
# perform binary encoding for categorical variable

# this function take in a pair of train and test data set, and the feature that need to be encode.

# it returns the two dataset with input feature encoded in binary representation

# this function assumpt that the feature to be encoded is already been encoded in a numeric manner 

# ranging from 0 to n-1 (n = number of levels in the feature). 



def binary_encoding(train_df, test_df, feat):

    # calculate the highest numerical value used for numeric encoding

    train_feat_max = train_df[feat].max()

    test_feat_max = test_df[feat].max()

    if train_feat_max > test_feat_max:

        feat_max = train_feat_max

    else:

        feat_max = test_feat_max

        

    # use the value of feat_max+1 to represent missing value

    train_df.loc[train_df[feat] == -1, feat] = feat_max + 1

    test_df.loc[test_df[feat] == -1, feat] = feat_max + 1

    

    # create a union set of all possible values of the feature

    union_val = np.union1d(train_df[feat].unique(), test_df[feat].unique())



    # extract the highest value from from the feature in decimal format.

    max_dec = union_val.max()

    

    # work out how the ammount of digtis required to be represent max_dev in binary representation

    max_bin_len = len("{0:b}".format(max_dec))

    index = np.arange(len(union_val))

    columns = list([feat])

    

    # create a binary encoding feature dataframe to capture all the levels for the feature

    bin_df = pd.DataFrame(index=index, columns=columns)

    bin_df[feat] = union_val

    

    # capture the binary representation for each level of the feature 

    feat_bin = bin_df[feat].apply(lambda x: "{0:b}".format(x).zfill(max_bin_len))

    

    # split the binary representation into different bit of digits 

    splitted = feat_bin.apply(lambda x: pd.Series(list(x)).astype(np.uint8))

    splitted.columns = [feat + '_bin_' + str(x) for x in splitted.columns]

    bin_df = bin_df.join(splitted)

    

    # merge the binary feature encoding dataframe with the train and test dataset - Done! 

    train_df = pd.merge(train_df, bin_df, how='left', on=[feat])

    test_df = pd.merge(test_df, bin_df, how='left', on=[feat])

    return train_df, test_df
# This function late in a list of features 'cols' from train and test dataset, 

# and performing frequency encoding. 

def count_encoding(cols, train_df, test_df):

    # we are going to store our new dataset in these two resulting datasets

    result_train_df=pd.DataFrame()

    result_test_df=pd.DataFrame()

    

    # loop through each feature column to do this

    for col in cols:

        

        # capture the frequency of a feature in the training set in the form of a dataframe

        col_freq=col+'_freq'

        freq=train_df[col].value_counts()

        freq=pd.DataFrame(freq)

        freq.reset_index(inplace=True)

        freq.columns=[[col,col_freq]]



        # merge ths 'freq' datafarme with the train data

        temp_train_df=pd.merge(train_df[[col]], freq, how='left', on=col)

        temp_train_df.drop([col], axis=1, inplace=True)



        # merge this 'freq' dataframe with the test data

        temp_test_df=pd.merge(test_df[[col]], freq, how='left', on=col)

        temp_test_df.drop([col], axis=1, inplace=True)



        # if certain levels in the test dataset is not observed in the train dataset, 

        # we assign frequency of zero to them

        temp_test_df.fillna(0, inplace=True)

        temp_test_df[col_freq]=temp_test_df[col_freq].astype(np.int32)



        if result_train_df.shape[0]==0:

            result_train_df=temp_train_df

            result_test_df=temp_test_df

        else:

            result_train_df=pd.concat([result_train_df, temp_train_df],axis=1)

            result_test_df=pd.concat([result_test_df, temp_test_df],axis=1)

    

    return result_train_df, result_test_df
cat_cols = ["supplier", "bracket_pricing", "material_id", "end_a_1x", 

            "end_a_2x", "end_x_1x", "end_x_2x", "end_a", "end_x", 

            "same_end_form", "component_id_1", "component_id_2", "component_id_3", 

            "component_id_4", "component_id_5", "component_id_6", 

            "component_id_7", "component_id_8"]
#do binary encoding for each category

train_set_df_bin_encode = train_set_df.copy()

test_set_df_bin_encode = test_set_df.copy()





for col in cat_cols:

    print("is handling {}".format(col))

    

    train_set_df_bin_encode[col].replace(np.nan,' ', regex=True, inplace= True)

    test_set_df_bin_encode[col].replace(np.nan,' ', regex=True, inplace= True)

    

    le = LabelEncoder()

    le.fit(list(train_set_df_bin_encode[col]) + list(test_set_df_bin_encode[col]))

    train_set_df_bin_encode[col] = le.fit_transform(train_set_df_bin_encode[col])

    test_set_df_bin_encode[col] = le.fit_transform(test_set_df_bin_encode[col])

    

    train_set_df_bin_encode, test_set_df_bin_encode = binary_encoding(train_set_df_bin_encode, test_set_df_bin_encode, col)

    

    train_set_df_bin_encode.drop(col, axis=1, inplace=True)

    test_set_df_bin_encode.drop(col, axis=1, inplace=True)
#do binary encoding for categories



count_train, count_test = count_encoding(cat_cols, train_set_df, test_set_df)



train_set_df_count_encode=pd.concat([train_set_df, count_train], axis=1)

test_set_df_count_encode=pd.concat([test_set_df,count_test], axis=1)



train_set_df_count_encode.drop(cat_cols, axis=1, inplace=True)

test_set_df_count_encode.drop(cat_cols, axis=1, inplace=True)
from sklearn.preprocessing import OneHotEncoder
train_set_df_ohot_encode = train_set_df.copy()

test_set_df_ohot_encode = test_set_df.copy()
le = LabelEncoder()



for col in cat_cols:

    train_set_df_ohot_encode[col].replace(np.nan,' ', regex=True, inplace= True)

    test_set_df_ohot_encode[col].replace(np.nan,' ', regex=True, inplace= True)
train_set_df_ohot_encode[cat_cols] = train_set_df_ohot_encode[cat_cols].apply(le.fit_transform)

test_set_df_ohot_encode[cat_cols] = test_set_df_ohot_encode[cat_cols].apply(le.fit_transform)



train_set_df_ohot_encode.fillna(0, inplace=True)

test_set_df_ohot_encode.fillna(0, inplace=True)
enc = OneHotEncoder()
train_set_df_ohot_encode = enc.fit_transform(train_set_df_ohot_encode)

test_set_df_ohot_encode = enc.fit_transform(test_set_df_ohot_encode)
print("original data shape is: {}".format(train_set_df.shape))

print("binary encode data shape is: {}".format(train_set_df_bin_encode.shape))

print("count encode data shape is: {}".format(train_set_df_count_encode.shape))

print("one hot encode data shape is: {}".format(train_set_df_ohot_encode.shape))
train_set_df_bin_encode.head()
train_set_df_count_encode.head()
#binary, count, onehot, binary and count

choose = "binary and count"





####################################################

if choose == "binary":

    train_set_df = train_set_df_bin_encode.copy()

    test_set_df = test_set_df_bin_encode.copy() 

    

elif choose == "count":

    train_set_df = train_set_df_count_encode.copy()

    test_set_df = test_set_df_count_encode.copy()

    

elif choose == "onehot":

    train_set_df = train_set_df_ohot_encode.copy()

    test_set_df = test_set_df_ohot_encode.copy()

    

elif choose == "binary and count":

    #find columns to keep

    cols_to_keep = train_set_df_count_encode.columns[train_set_df_count_encode.columns.str.endswith("freq")]

    #frame with useful columns

    train_part1 = train_set_df_count_encode[cols_to_keep]

    test_part1 = test_set_df_count_encode[cols_to_keep]

    #merge frames together

    train_set_df = pd.merge(train_set_df_bin_encode, train_part1, left_index=True, right_index=True)

    test_set_df = pd.merge(test_set_df_bin_encode, test_part1, left_index=True, right_index=True)

    

else:

    print("please choose a encode method.")
#fill out all null by 0



train_set_df.fillna(0, inplace=True)

test_set_df.fillna(0, inplace=True)
train_set_df.shape
train_set_df.head(10)
train_set_df.tail(10)
train_set_df.columns.tolist()
train_set_df['annual_usage'].describe()
bill_of_materials_df.head(20)
bill_of_materials_df.info(null_counts=True)
dummy_df=train_set_df.copy()
dummy_df.head(5)
dummy_df.loc[0:2, 'annual_usage']=5
dummy_df.iloc[0:2,0:2]
bill_of_materials_df['component_id_1'].nunique()
bill_of_materials_df['component_id_1'].value_counts()
train_set_df["cost"] = train_label



data = train_set_df.copy()
#define a evaluation function



def rmsle_score(preds, true):

    rmsle_score = (np.sum((np.log1p(preds)-np.log1p(true))**2)/len(true))**0.5

    return rmsle_score
#Define a evaluation matrix 

from sklearn.metrics.scorer import make_scorer



RMSLE = make_scorer(rmsle_score)
# split for cross_val_score machine learning model



label = "cost"



data_labels = data.columns.tolist()

data_labels.remove(label)



X = data[data_labels]

y = data[label]
#XGB Regression and KFold

import xgboost as xgb

from xgboost import XGBRegressor

from sklearn.model_selection import KFold



start = time.time()



xgb_regressor=XGBRegressor(max_depth=7, 

                           n_estimators=500, 

                           objective="reg:linear", 

                           min_child_weight = 6,

                           subsample = 0.87,

                           colsample_bytree = 0.50,

                           scale_pos_weight = 1.0,                       

                           learning_rate=0.1,

                          n_jobs=4)

scores = []



kf = KFold(n_splits=5)



for i, (train_index, test_index) in enumerate(kf.split(X, y)):

    #print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]

    y_train, y_test = y[train_index], y[test_index]

    

    y_log = np.log1p(y_train)



    model = xgb_regressor.fit(X_train, y_log, eval_metric=RMSLE)

    xgb_preds1 = model.predict(X_test)



    xgb_preds = np.expm1(xgb_preds1)

        

    rmsle_xgb = rmsle_score(xgb_preds, y_test)

    print ("Folder cv {}, XGB RMSLE is : {}".format(i+1, rmsle_xgb))

    scores.append(rmsle_xgb)

    

print("Mean RMSLE is : {}".format(np.mean(scores)))



end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
label = "cost"



data_labels = train_set_df.columns.tolist()

data_labels.remove(label)



train_df = train_set_df[data_labels]

train_label = train_set_df[label]



test = test_set_df.copy()
#XGB regression



start = time.time()

xgb_regressor=XGBRegressor(max_depth=7, 

                           n_estimators=500, 

                           objective="reg:linear", 

                           min_child_weight = 6,

                           subsample = 0.87,

                           colsample_bytree = 0.50,

                           scale_pos_weight = 1.0,                       

                           learning_rate=0.1)



label_log=np.log1p(train_label)



model=xgb_regressor.fit(train_df, label_log)

xgb_preds=model.predict(test)



xgb_preds=np.expm1(xgb_preds)







end = time.time()

duration = end - start

print ("It takes {} seconds".format(duration))
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission.cost = xgb_preds



#sample_submission.to_csv("../output/submission.csv", index=False)

sample_submission.head()