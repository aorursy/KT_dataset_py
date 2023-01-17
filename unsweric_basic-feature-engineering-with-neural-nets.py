import numpy as np

import pandas as pd

data_train=pd.read_csv('train.csv')

target_data=data_train['SalePrice']

data_train=data_train.drop('SalePrice', axis = 1)





data_test=pd.read_csv('test.csv')

combine=[data_train,data_test]

data = pd.concat(combine)
data.head()
data.info()
data.describe(include=['O'])
raw_data=data.drop('Id', axis = 1)
#drop "Alley" as it has only 91 entries

raw_data=raw_data.drop('Alley', axis = 1)
# replace miss values of 'LotFrontage' with its median

raw_data.loc[ raw_data.LotFrontage.isnull(),'LotFrontage'] =raw_data['LotFrontage'].median()

     
# replace miss values of 'MasVnrType' with its mode

print(raw_data['MasVnrType'].mode())

raw_data.loc[ raw_data.MasVnrType.isnull(),'MasVnrType'] ='None'
# replace miss values of 'MasVnrArea' with its mode

print(raw_data['MasVnrArea'].mode())

raw_data.loc[ raw_data.MasVnrArea.isnull(),'MasVnrArea'] =0.0

# replace miss values of 'BsmtQual' with its mode

print(raw_data['BsmtQual'].mode())

raw_data.loc[ raw_data.BsmtQual.isnull(),'BsmtQual'] ='TA'
# replace miss values of 'BsmtCond' with its mode

print(raw_data['BsmtCond'].mode())

raw_data.loc[ raw_data.BsmtCond.isnull(),'BsmtCond'] ='TA'
# replace miss values of 'BsmtExposure' with its mode

print(raw_data['BsmtExposure'].mode())

raw_data.loc[ raw_data.BsmtExposure.isnull(),'BsmtExposure'] ='No'
# replace miss values of 'BsmtFinType1' with its mode

print(raw_data['BsmtFinType1'].mode())

raw_data.loc[ raw_data.BsmtFinType1.isnull(),'BsmtFinType1'] ='Unf'
# replace miss values of 'BsmtFinType2' with its mode

print(raw_data['BsmtFinType2'].mode())

raw_data.loc[ raw_data.BsmtFinType2.isnull(),'BsmtFinType2'] ='Unf'
# replace miss values of 'BElectrical' with its mode

print(raw_data['Electrical'].mode())

raw_data.loc[ raw_data.Electrical.isnull(),'Electrical'] ='SBrkr'
#drop "FireplaceQu" as it has only 770 entries

raw_data=raw_data.drop('FireplaceQu', axis = 1)
# replace miss values of 'GarageType' with its mode

print(raw_data['GarageType'].mode())

raw_data.loc[ raw_data.GarageType.isnull(),'GarageType'] ='Attchd'
# replace miss values of 'GarageYrBlt' with its mode

print(raw_data['GarageYrBlt'].mode())

raw_data.loc[ raw_data.GarageYrBlt.isnull(),'GarageYrBlt'] =2005.0
# replace miss values of 'GarageQual' with its mode

print(raw_data['GarageQual'].mode())

raw_data.loc[ raw_data.GarageQual.isnull(),'GarageQual'] ='TA'
# replace miss values of 'GarageFinish' with its mode

print(raw_data['GarageFinish'].mode())

raw_data.loc[ raw_data.GarageFinish.isnull(),'GarageFinish'] ='Unf'
# replace miss values of 'GarageCond' with its mode

print(raw_data['GarageCond'].mode())

raw_data.loc[ raw_data.GarageCond.isnull(),'GarageCond'] ='TA'
#drop "PoolQC " as it has only 7 entries

raw_data=raw_data.drop('PoolQC', axis = 1)
#drop "Fence " as it has only 281 entries

raw_data=raw_data.drop('Fence', axis = 1)
#drop "MiscFeature  " as it has only 54 entries

raw_data=raw_data.drop('MiscFeature', axis = 1)
# replace miss values of 'BsmtFinSF1' with its mode

print(raw_data['BsmtFinSF1'].mode())

raw_data.loc[ raw_data.BsmtFinSF1.isnull(),'BsmtFinSF1'] =0.0
# replace miss values of 'BsmtFinSF2' with its mode

print(raw_data['BsmtFinSF2'].mode())

raw_data.loc[ raw_data.BsmtFinSF2.isnull(),'BsmtFinSF2'] =0.0
# replace miss values of 'BsmtFullBath' with its mode

print(raw_data['BsmtFullBath'].mode())

raw_data.loc[ raw_data.BsmtFullBath.isnull(),'BsmtFullBath'] =0.0
# replace miss values of 'BsmtHalfBath' with its mode

print(raw_data['BsmtHalfBath'].mode())

raw_data.loc[ raw_data.BsmtHalfBath.isnull(),'BsmtHalfBath'] =0.0
# replace miss values of 'BsmtUnfSF' with its mode

print(raw_data['BsmtUnfSF'].mode())

raw_data.loc[ raw_data.BsmtUnfSF.isnull(),'BsmtUnfSF'] =0.0
# replace miss values of 'Exterior1st' with its mode

print(raw_data['Exterior1st'].mode())

raw_data.loc[ raw_data.Exterior1st.isnull(),'Exterior1st'] ='VinylSd'
# replace miss values of 'Exterior2nd ' with its mode

print(raw_data['Exterior2nd'].mode())

raw_data.loc[ raw_data.Exterior2nd .isnull(),'Exterior2nd'] ='VinylSd'
# replace miss values of 'Functional' with its mode

print(raw_data['Functional'].mode())

raw_data.loc[ raw_data.Functional.isnull(),'Functional'] ='Typ'
# replace miss values of 'GarageArea' with its mode

print(raw_data['GarageArea'].mode())

raw_data.loc[ raw_data.GarageArea.isnull(),'GarageArea'] =0.0
# replace miss values of 'GarageCars' with its mode

print(raw_data['GarageCars'].mode())

raw_data.loc[ raw_data.GarageCars.isnull(),'GarageCars'] =2.0
# replace miss values of 'GarageCars' with its mode

print(raw_data['GarageCars'].mode())

raw_data.loc[ raw_data.GarageCars.isnull(),'GarageCars'] =2.0
# replace miss values of 'KitchenQual' with its mode

print(raw_data['KitchenQual'].mode())

raw_data.loc[ raw_data.KitchenQual.isnull(),'KitchenQual'] ='TA'
# replace miss values of 'MSZoning' with its mode

print(raw_data['MSZoning'].mode())

raw_data.loc[ raw_data.MSZoning.isnull(),'MSZoning'] ='RL'
# replace miss values of 'SaleType' with its mode

print(raw_data['SaleType'].mode())

raw_data.loc[ raw_data.SaleType.isnull(),'SaleType'] ='WD'
# replace miss values of 'TotalBsmtSF' with its mode

print(raw_data['TotalBsmtSF'].mode())

raw_data.loc[ raw_data.TotalBsmtSF.isnull(),'TotalBsmtSF'] =0.0
# replace miss values of 'Utilities' with its mode

print(raw_data['Utilities'].mode())

raw_data.loc[ raw_data.Utilities.isnull(),'Utilities'] ='AllPub'
raw_data.info()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

numerical = list(raw_data.select_dtypes(include=['int64']).columns.values)+list(raw_data.select_dtypes(include=['float64']).columns.values)

print(numerical)

raw_data[numerical] = scaler.fit_transform(raw_data[numerical])
raw_data.head()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



categorical=list(raw_data.select_dtypes(include=['object']).columns.values)

print(categorical)

for value in categorical:

    le.fit(raw_data[value])

    raw_data[value]=le.transform(raw_data[value])

raw_data.head()

from sklearn.cross_validation import train_test_split

feature_train_all=raw_data[:1460]

feature_test=raw_data[1460:]

X_train, X_test, y_train, y_test = train_test_split(feature_train_all,target_data, test_size = 0.2, random_state = 0)

print(X_train.shape[0])
import tensorflow as tf

import math



# Parameters



training_epochs = 500

keep_probability = 1

#learning_rate = 0.001

keep_prob = tf.placeholder(tf.float32)







x = tf.placeholder("float", [X_train.shape[0], X_train.shape[1]])

weight_1=tf.Variable(tf.truncated_normal([X_train.shape[1],500], mean=0, stddev=1))

bias_1=tf.Variable(tf.zeros(500))

layer_1 = tf.add(tf.matmul(x, weight_1),bias_1)

layer_1 = tf.nn.dropout(layer_1, keep_prob)



weight_2=tf.Variable(tf.truncated_normal([layer_1.get_shape().as_list()[1],500], mean=0, stddev=1))

bias_2=tf.Variable(tf.zeros(500))

layer_2 = tf.add(tf.matmul(layer_1, weight_2),bias_2)

layer_2 = tf.nn.dropout(layer_2, keep_prob)



weight_3=tf.Variable(tf.truncated_normal([layer_2.get_shape().as_list()[1],500], mean=0, stddev=1))

bias_3=tf.Variable(tf.zeros(500))

layer_3 = tf.add(tf.matmul(layer_2, weight_3),bias_3)

layer_3 = tf.nn.dropout(layer_3, keep_prob)



weight_4=tf.Variable(tf.truncated_normal([layer_3.get_shape().as_list()[1],1], mean=0, stddev=1))

bias_4=tf.Variable(tf.zeros(1))

output = tf.add(tf.matmul(layer_3, weight_4),bias_4)





# Mean squared error

y = tf.placeholder("float", [1168])

cost = tf.reduce_sum(tf.pow(output-y, 2))/1168

optimizer = tf.train.AdamOptimizer().minimize(cost)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)



#predict feature train

x_train=tf.placeholder("float", [292, 74])

L1 = tf.add(tf.matmul(x_train, weight_1),bias_1)

L2=tf.add(tf.matmul(L1, weight_2),bias_2)

L3 = tf.add(tf.matmul(L2, weight_3),bias_3)

Pred_train= tf.add(tf.matmul(L3, weight_4),bias_4)



#predict feature original

x_original=tf.placeholder("float", [1168, 74])

L1 = tf.add(tf.matmul(x_original, weight_1),bias_1)

L2=tf.add(tf.matmul(L1, weight_2),bias_2)

L3 = tf.add(tf.matmul(L2, weight_3),bias_3)

Pred_original= tf.add(tf.matmul(L3, weight_4),bias_4)



#predict feature test

x_test=tf.placeholder("float", [1459, 74])

L1 = tf.add(tf.matmul(x_test, weight_1),bias_1)

L2=tf.add(tf.matmul(L1, weight_2),bias_2)

L3 = tf.add(tf.matmul(L2, weight_3),bias_3)

Pred_test= tf.add(tf.matmul(L3, weight_4),bias_4)



# Initializing the variables

init = tf.global_variables_initializer()





# Launch the graph

with tf.Session() as sess:

    sess.run(init)

    # Training cycle

    for epoch in range(training_epochs):

        sess.run(optimizer, feed_dict={x: X_train, y: y_train, keep_prob: keep_probability})

    sess.run(Pred_train, feed_dict={x_train: X_test})  

    sess.run(Pred_test, feed_dict={x_test: feature_test})

    sess.run(Pred_original, feed_dict={x_original: X_train})

    #transfor Pred_train to numpy array

    Pred_train_np = Pred_train.eval({x_train: X_test})

    #transfor Pred_test to numpy array

    Pred_test_np = Pred_test.eval({x_test: feature_test})

    #transfor Pred_original to numpy array

    Pred_original_np = Pred_original.eval({x_original: X_train})
from sklearn.metrics import r2_score

Pred_train_dataframe= pd.DataFrame(Pred_train_np, index=y_test.index)

Pred_original_dataframe= pd.DataFrame(Pred_original_np, index=y_train.index)

print(r2_score(y_test,Pred_train_dataframe))

print(r2_score(y_train,Pred_original_dataframe))
df = pd.DataFrame(Pred_train_np, index=y_test.index)
print(type(Pred_train_np))

print(Pred_train_np.shape)

print(Pred_test_np.shape)

print(Pred_train_np)

print(type(y_test))
df = pd.DataFrame(Pred_train_np, index=y_test.index)

print(df)

print(y_test)
from sklearn.metrics import r2_score

print(r2_score(y_test,df))
submission = pd.DataFrame({

        "Id": data_test["Id"],

        "SalePrice": Y_pred

    })

submission.to_csv('submission.csv', index=False)