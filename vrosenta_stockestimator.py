import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow import estimator
import pandas as pd
tf.reset_default_graph()
df = pd.read_csv('../input/data.csv') # read data set using pandas
df = df.drop(['Date'],axis=1) # Drop Date feature
df = df.dropna(inplace=False)  # Remove all nan entries.
df = df.drop(['Adj Close','Volume'],axis=1) # Drop Adj close and volume feature
df_train = df[:1059]    # 60% training data and 40% testing data
df_test = df[1059:]
def normalize(df):    
    return (df - df.min()) / (df.max() - df.min())

def denormalize(df,norm_data):    
    return (norm_data * (df.max() - df.min())) + df.min()

# We want to predict Close value of stock 

X_train = normalize(df_train.drop(['Close'],axis=1)).values
y_train = normalize(df_train['Close']).values

#X_train.all
# y is output and x is features.

X_test = normalize(df_test.drop(['Close'],axis=1)).values
y_test = normalize(df_test['Close']).values
feat_cols=[tf.feature_column.numeric_column('x', shape=[3])]
deep_model=tf.estimator.DNNRegressor(feature_columns=feat_cols, 
                                   hidden_units=[4,4], 
                                   activation_fn=tf.nn.relu)
input_fn = estimator.inputs.numpy_input_fn(x={'x':X_train}, 
                                           y=y_train,
                                           shuffle= True,
                                           num_epochs=5000,
                                           batch_size=200
                                          )
train = deep_model.train(input_fn=input_fn, steps=5000)
train = tf.train.GradientDescentOptimizer(0.001)
input_fn_eval = estimator.inputs.numpy_input_fn( x = {'x':X_test},
                                                   
                                                   shuffle = False)
preds=list(deep_model.predict(input_fn=input_fn_eval))
predictions = [p['predictions'][0] for p in preds]
pred = np.asarray(predictions)
pred = denormalize(df_test['Close'],pred)
y_test = denormalize(df_test['Close'],y_test)
plt.plot(range(y_test.shape[0]),y_test,label="Original Data")
plt.plot(range(y_test.shape[0]),pred,label="Predicted Data")
plt.legend(loc='best')
plt.ylabel('Stock Value')
plt.xlabel('Days')
plt.title('Stock Market Nifty')
plt.show()
#print(y_test , '\n\n',y_test)




