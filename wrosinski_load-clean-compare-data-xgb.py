# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import gc





from sklearn.cross_validation import KFold

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder

from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score

from sklearn import preprocessing



from scipy.sparse import csr_matrix, hstack



import warnings

warnings.filterwarnings("ignore")
numerical_cols = ['ncodpers', 'age', 'antiguedad', 'renta']





feature_cols = ['ind_actividad_cliente', 

                "ind_empleado", "pais_residencia" ,"sexo" , "ind_nuevo", 

                 "nomprov", "segmento", 'indrel', 'tiprel_1mes', 'indresi', 'indext',

               'conyuemp', 'indfall', 'canal_entrada']



dtype_list = {'ind_cco_fin_ult1': 'float16', 'ind_deme_fin_ult1': 'float16', 'ind_aval_fin_ult1': 'float16', 'ind_valo_fin_ult1': 'float16', 'ind_reca_fin_ult1': 'float16', 'ind_ctju_fin_ult1': 'float16', 'ind_cder_fin_ult1': 'float16', 'ind_plan_fin_ult1': 'float16', 'ind_fond_fin_ult1': 'float16', 'ind_hip_fin_ult1': 'float16', 'ind_pres_fin_ult1': 'float16', 'ind_nomina_ult1': 'float16', 'ind_cno_fin_ult1': 'float16', 'ncodpers': 'int64', 'ind_ctpp_fin_ult1': 'float16', 'ind_ahor_fin_ult1': 'float16', 'ind_dela_fin_ult1': 'float16', 'ind_ecue_fin_ult1': 'float16', 'ind_nom_pens_ult1': 'float16', 'ind_recibo_ult1': 'float16', 'ind_deco_fin_ult1': 'float16', 'ind_tjcr_fin_ult1': 'float16', 'ind_ctop_fin_ult1': 'float16', 'ind_viv_fin_ult1': 'float16', 'ind_ctma_fin_ult1': 'float16'}

target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',

       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',

       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',

       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',

       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',

       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',

       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',

       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']



data_path = "../input/"

train_file = data_path + "train_ver2.csv"

test_file = data_path + "test_ver2.csv"

train_size = 13647309

nrows = 2000000 # change this value to read more rows from train

start_index = train_size - nrows
class SantanderLoader(object):

    

    def __init__(self, train, test):

        self.train_file = train

        self.test_file = test

        

        

    def load_categorical(self, feature_cols ):

        

        start_index = train_size - nrows	

        

        for ind, col in enumerate(feature_cols):

            print(col)

            train = pd.read_csv(self.train_file, usecols=[col])

            test = pd.read_csv(self.test_file, usecols=[col])

            train.fillna(-1, inplace=True)

            test.fillna(-1, inplace=True)

            if train[col].dtype == "object":

                le = LabelEncoder()

                le.fit(list(train[col].values) + list(test[col].values))

                temp_train_X = le.transform(list(train[col].values)).reshape(-1,1)[start_index:,:]

                temp_test_X = le.transform(list(test[col].values)).reshape(-1,1)

            else:

                temp_train_X = np.array(train[col]).reshape(-1,1)[start_index:,:]

                temp_test_X = np.array(test[col]).reshape(-1,1)

            if ind == 0:

                train_X = temp_train_X.copy()

                test_X = temp_test_X.copy()

            else:

                train_X = np.hstack([train_X, temp_train_X])

                test_X = np.hstack([test_X, temp_test_X])

            print(train_X.shape, test_X.shape)

        del train

        del test

        print ("Categorical features loaded.")

        return train_X, test_X

        

        

    def load_numeric(self, numeric_cols ):

        

        start_index = train_size - nrows	

        

        for ind, col in enumerate(numerical_cols):

            print(col)

            train = pd.read_csv(self.train_file, usecols=[col])

            test = pd.read_csv(self.test_file, usecols=[col])

            if train[col].dtype == "object":

                temp_train_X = pd.to_numeric(train[col], 'coerce').fillna(-1).astype('float64').reshape(-1,1)[start_index:,:]

                temp_test_X = pd.to_numeric(test[col], 'coerce').fillna(-1).astype('float64').reshape(-1,1)

            else:

                temp_train_X = np.array(pd.to_numeric(train[col], 'coerce').fillna(-999).astype('float64')).reshape(-1,1)[start_index:,:]

                temp_test_X = np.array(pd.to_numeric(test[col], 'coerce').fillna(-999).astype('float64')).reshape(-1,1)

            if ind == 0:

                train_X_f = temp_train_X.copy()

                test_X_f = temp_test_X.copy()

            else:

                train_X_f = np.hstack([train_X_f, temp_train_X])

                test_X_f = np.hstack([test_X_f, temp_test_X])

        print ("Numeric features loaded.")

        return train_X_f, test_X_f

    

    

    def load_dates(self, ):

        

        start_index = train_size - nrows	

        

        train_X_d = pd.read_csv(self.train_file, usecols = ['fecha_dato', 'fecha_alta'], nrows = nrows)

        test_X_d = pd.read_csv(self.test_file, usecols = ['fecha_dato', 'fecha_alta'])



        print ("Date features loaded")

        return train_X_d, test_X_d

    

    

    def stack_features(self, cats, nums, dates):

        

        cats_df = pd.DataFrame(cats)

        nums_df = pd.DataFrame(nums)

        dates_df = pd.DataFrame(dates)

        

        stacked = pd.concat((cats_df, nums_df, dates_df), axis = 1)

        print ("Columns stacked")

        return stacked

    

    

    def name_columns(self, data, set_index = False):

        

        #df = pd.DataFrame(data)

        data.columns = ['ind_actividad_cliente', 

                "ind_empleado", "pais_residencia" ,"sexo" , "ind_nuevo", 

                 "nomprov", "segmento", 'indrel', 'tiprel_1mes', 'indresi', 'indext',

               'conyuemp', 'indfall', 'canal_entrada',

                   'ncodpers', 'age', 'antiguedad', 'renta', 

                   'fecha_dato', 'fecha_alta']

        

        if set_index:

            data.set_index(['ncodpers'], inplace = True)

            print ("Index set to ncodpers")

        

        print ("Columns named")

        return data

    

    

    def to_csv(self, data):

        

        data.to_csv('data.csv', index = False)

        print ("File exported to csv")

        

        

    def label_loader(self, drop_date = False, set_index = False):

        

        full_y = pd.read_csv(self.train_file, usecols=['fecha_dato'] + ['ncodpers'] + target_cols, 

                             dtype=dtype_list, nrows = nrows)

        full_y.fillna(0, inplace = True)

        

        if drop_date:

            full_y.drop(['fecha_dato'], axis = 1, inplace = True)

            print ("Date dropped")

            

        if set_index:

            full_y.set_index(['ncodpers'], inplace = True)

            print ("Index set to ncodpers")

            

        

        print ("Labels loaded")

        return full_y
loader = SantanderLoader(train_file, test_file)



train1, test1 = loader.load_categorical(feature_cols)

train2, test2 = loader.load_numeric(numerical_cols)

train3, test3 = loader.load_dates()



stacked_train = loader.stack_features(train1, train2, train3)

named_train = loader.name_columns(stacked_train)

stacked_test = loader.stack_features(test1, test2, test3)

named_test = loader.name_columns(stacked_test)



# loader.to_csv(named_train) <- if you'd like to save the output for further processing,

# without needing to load the raw data each time.



print (named_train.info(memory_usage = True))



del train1, test1, train2, test2, train3, test3, stacked_train, stacked_test

gc.collect()



y = loader.label_loader()

print (y.head())

class SantanderCleaner(object):

    

    def __init__(self, data, labels):

        self.data = data

        self.labels = labels



    def month(self, year, month):

        

        monthly_data = self.data[self.data['fecha_dato'] == "201{}-0{}-28".format(year, month) ]

        monthly_labels = self.labels[self.labels['fecha_dato'] == "201{}-0{}-28".format(year, month) ]

        

        print ("Month {} from year {} data shape: ".format(year, month), monthly_data.shape)

        return monthly_data, monthly_labels

    

    def get_same(self, data1, data2):

        

        data_1st = data1[data1.ncodpers.isin(data2.ncodpers.values)]

        data_2nd = data2[data2.ncodpers.isin(data1.ncodpers.values)]

        

        print ("Shape when having same clients: ", data_1st.shape)

        return data_1st, data_2nd

    

    def prepare_to_train(self, data1, labels, ncod_delete = True):

        

        if ncod_delete == True:

            data2 = data1.drop(['ncodpers', 'fecha_dato', 'fecha_alta'], 1)

            labels1 = labels.drop(['ncodpers', 'fecha_dato'], 1)

            labels1 = labels1.astype(int)

        else:

            data2 = data1.drop(['fecha_dato', 'fecha_alta'], 1)

            labels1 = labels.drop(['fecha_dato'], 1)

            labels1 = labels1.astype(int)



        print ("Final data shape: ", data2.shape, "Final labels shape:", labels1.shape)

        return data2, labels1

    



    def impute_missing(self, data1):



        imputer_cat = preprocessing.Imputer(-1, 'most_frequent', 0)

        imputer_renta = preprocessing.Imputer(-999, 'mean', 0)

        

        renta = data1.iloc[:, 16:]

        cat_data = data1.iloc[:, :16]



        cat_data = imputer_cat.fit_transform(cat_data)

        renta = imputer_renta.fit_transform(renta)



        imputed_data = pd.DataFrame(np.hstack([cat_data, renta]))



        print ("Data Imputed")

        return imputed_data



    def numerical_scale(self, data1):



        scaler = RobustScaler()

        data1.loc[:,16]= scaler.fit_transform(data1.loc[:,16])



        print ("Numerical data scaled")

        return data1



training = SantanderCleaner(named_train, y)



january, january_y = training.month(5, 1)

february, february_y = training.month(5, 2)



january, january_y  = training.prepare_to_train(january, january_y )

february, february_y = training.prepare_to_train(february, february_y)



january = training.impute_missing(january)

february = training.impute_missing(february)



january = training.numerical_scale(january)

february = training.numerical_scale(february)





# For validation



march, march_y = training.month(5, 3)

march, march_y = training.prepare_to_train(march, march_y)

march = training.impute_missing(march)

march = training.numerical_scale(march)



#del full_df_train, full_y

#gc.collect()



test = SantanderCleaner(named_test, _)

test_data = named_test.drop(['ncodpers', 'fecha_dato', 'fecha_alta'], 1)

test_data = test.impute_missing(test_data)

test_data = test.numerical_scale(test_data)
train = pd.concat([january, february], axis = 0)

labels = pd.concat([january_y, february_y], axis = 0)



train2 = train.iloc[:100000, :]

labels2 = labels.iloc[:100000, :]



X_train, X_val, y_train, y_val = train_test_split(train2, labels2, test_size = 0.2, 

                                                  random_state = 669)



print ( type(labels2) )
from sklearn.calibration import CalibratedClassifierCV

from sklearn.multiclass import OneVsRestClassifier

from xgboost import XGBClassifier



nb_classes = labels.shape[1]



params = {

    'n_estimators': 10,

    'max_depth': 10,

}



xgbc = XGBClassifier(**params)





ova_xgbc = OneVsRestClassifier(xgbc)

ova_xgbc.fit(X_train, y_train)

ova_preds = ova_xgbc.predict(X_val)







F1 = f1_score(y_val, ova_preds, average = "macro")

print("F1 score: ", F1 )
last_instance_df = y.drop_duplicates('ncodpers', keep='last')

last_instance_df = last_instance_df.drop(['fecha_dato'], 1)



preds = ova_xgbc.predict(test_data)

print ("Test set predictions done.", '\n')

print ("Shape of test predictions: ", preds.shape, '\n')







print("Getting last instance dict..", '\n')

last_instance_df = last_instance_df.fillna(0).astype('int')

cust_dict = {}

target_cols = np.array(target_cols)

for ind, row in last_instance_df.iterrows():

    cust = row['ncodpers']

    used_products = set(target_cols[np.array(row[1:])==1])

    cust_dict[cust] = used_products



    

print("Creating submission..")

preds = np.argsort(preds, axis=1)

preds = np.fliplr(preds)

test_id = np.array(pd.read_csv(test_file, usecols=['ncodpers'])['ncodpers'])

final_preds = []

for ind, pred in enumerate(preds):

    cust = test_id[ind]

    top_products = target_cols[pred]

    used_products = cust_dict.get(cust,[])

    new_top_products = []

    for product in top_products:

        if product not in used_products:

            new_top_products.append(product)

        if len(new_top_products) == 7:

            break

    final_preds.append(" ".join(new_top_products))



len(final_preds[0])

len(final_preds)

final_preds[0]
out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})

out_df.to_csv('XGBoostClassifier.csv', index=False)