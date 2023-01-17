# import os
# i = 0
# for dirname, _, filenames in os.walk('/kaggle/input'):  
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#         i += 1
#         if i > 10:
#             break
import pydicom
import os
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
im_path = "../input/osic-pulmonary-fibrosis-progressiont/"
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

print('Training data shape: ', train_df.shape)
train_df.head()
train_df = pd.get_dummies(train_df, columns=['Sex'])
train_df = pd.get_dummies(train_df, columns=['SmokingStatus'])
train_df = train_df.rename(columns={"Sex_Female": "Female", 
                                    "Sex_Male": "Male",
                                    "SmokingStatus_Currently smokes": "CurrentlySmokes",
                                    "SmokingStatus_Ex-smoker": "ExSmoker",
                                    "SmokingStatus_Never smoked": "NeverSmoked"})
train_df.head()
X = train_df.drop(['Patient','FVC'], axis=1)
y = train_df['FVC']
# test_df = pd.get_dummies(test_df, columns=['Sex'])
# test_df = pd.get_dummies(test_df, columns=['SmokingStatus'])
# test_df = test_df.rename(columns={  "Sex_Male": "Male",
#                                     "SmokingStatus_Ex-smoker": "ExSmoker",
#                                     "SmokingStatus_Never smoked": "NeverSmoked"})
# test_df.insert(5, 'Female', np.zeros(5))
# test_df.insert(7,'CurrentlySmokes',np.zeros(5))
# test_df.head()
# Splite data into training and testing
from sklearn import model_selection

# Reserve 20% for testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

print('training data has ' + str(X_train.shape[0]) + 
      ' observation with ' + str(X_train.shape[1]) + ' features')
print('test data has ' + str(X_test.shape[0]) + 
      ' observation with ' + str(X_test.shape[1]) + ' features')
# standardization (x-mean)/std
# normalization (x-x_min)/(x_max-x_min) ->[0,1]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
import xgboost as xgb
from xgboost import XGBRegressor
regr_XGB = XGBRegressor()
regr_XGB.fit(X_train, y_train)
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
cv_score = model_selection.cross_val_score(regr_XGB, X_train, y_train, cv=5)
print(cv_score)

# Possible hyperparamter options for XGBoost
# Choose the number of trees, max depth and other
parameters = {'max_depth': np.arange(5, 10),
              'colsample_bytree': np.arange(0.6,1,0.1),
              'subsample': np.arange(0.6,1,0.1),
              'eta': np.logspace(-2, 0, 10)
              }
Grid_XGB = GridSearchCV(regr_XGB,parameters, cv=5)
Grid_XGB.fit(X_train, y_train)
best_XGB_model = Grid_XGB.best_estimator_
best_XGB_model
regr_XGB_opt = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.8999999999999999, eta=0.01,
             gamma=0, gpu_id=-1, importance_type='gain',
             interaction_constraints='', learning_rate=0.300000012,
             max_delta_step=0, max_depth=6, min_child_weight=1, missing=None,
             monotone_constraints='()', n_estimators=100, n_jobs=0,
             num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
             scale_pos_weight=1, subsample=0.7999999999999999,
             tree_method='exact', validate_parameters=1, verbosity=None)
regr_XGB_opt.fit(X_train, y_train)
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred,color = 'r', alpha = 0.3)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)], color = 'k')
plt.xlabel('FVC$_{\mathrm{test}}$')
plt.ylabel('FVC$_{\mathrm{pred}}$')
plt.rcParams.update({'font.size': 22})
from sklearn.metrics import mean_squared_error
mse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (mse**0.5))
data_dmatrix = xgb.DMatrix(data=X,label=y)
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10,
                    num_boost_round=200,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
print((cv_results["test-rmse-mean"]).tail(1))
importances = regr_XGB.feature_importances_
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature importance ranking by XGBoost Model:")
for ind in range(X.shape[1]):
    print ("%s : %.4f" %(X.columns[indices[ind]],importances[indices[ind]]))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    
    #sigma_clip = sigma + C1
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)

def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss

def make_model():
    z = L.Input((8,), name="Patient")
    x = L.Dense(128, activation="relu", name="d1")(z)
    x = L.Dense(512, activation="relu", name="d2")(x)
    #x = L.Dense(100, activation="relu", name="d3")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                     name="preds")([p1, p2])
    
    model = M.Model(z, preds, name="CNN")
    #model.compile(loss=qloss, optimizer="adam", metrics=[score])
    model.compile(loss=mloss(0.8), optimizer="adam", metrics=[score])
    return model
net = make_model()
print(net.summary())
print(net.count_params())
ROOT = "../input/osic-pulmonary-fibrosis-progression"
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
sub = sub.drop('Patient_Week', axis=1)
from sklearn.model_selection import KFold
NFOLD = 5
kf = KFold(n_splits=NFOLD)
%%time
cnt = 0
pred = np.zeros((X.shape[0], 3))
for tr_idx, val_idx in kf.split(X):
    cnt += 1
    print(f"FOLD {cnt}")
    net = make_model()
    
    net.fit(X[tr_idx], y[tr_idx], batch_size=50, epochs=500, 
            validation_data=(X[val_idx], y[val_idx]), verbose=0) 
    print("train", net.evaluate(X[tr_idx], y[tr_idx], verbose=0, batch_size=50))
    print("val", net.evaluate(X[val_idx], y[val_idx], verbose=0, batch_size=50))
    print("predict val...")
    pred[val_idx] = net.predict(X[val_idx], batch_size=50, verbose=0)
    print("predict test...")

from sklearn.metrics import mean_absolute_error
sigma_opt = mean_absolute_error(y, pred[:, 1])
unc = pred[:,2] - pred[:, 0]
sigma_mean = np.mean(unc)
print(sigma_opt, sigma_mean)
plt.figure(figsize=(5,5))
plt.scatter(y,pred[:,1], color = 'r', alpha = 0.3)
plt.plot([1000,6000],[1000,6000], color = 'k')
plt.xlabel('FVC$_{\mathrm{test}}$')
plt.ylabel('FVC$_{\mathrm{pred}}$')
plt.rcParams.update({'font.size': 22})

mse = np.sqrt(mean_squared_error(y, pred[:,1]))
print("RMSE: %f" % (mse**0.5))

idxs = np.random.randint(0, y.shape[0], 50)
plt.figure(figsize=(10,10))
plt.plot(y.values[idxs], label="ground truth")
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()
def lin_decay(w, a, b):
    return a * w + b

def exp_decay(w, r, b, c):
    return c * np.exp(-r * w) + b

def log_decay(w, r, b, c):
    return b / (1 + c * np.exp(-r * w))

patient_ids = os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/train')

FVC_list = []
week_list = []
pct_list = []

pars_lin, pcov_lin = [], []
pars_exp, pcov_exp = [], []
pars_log, pcov_log = [], []
RSS_lin = RSS_exp = RSS_log = 0

#fit_values = []
#columns = ['Patient', 'r', 'b']
    
for i in range(len(patient_ids)):
    pid = patient_ids[i]
    week = train_df.loc[train_df['Patient'] == pid]['Weeks']
    week = week - min(week)
    week_list.append(week)
    FVC = train_df.loc[train_df['Patient'] == pid]['FVC']
    FVC_list.append(FVC)
    pct = train_df.loc[train_df['Patient'] == pid]['Percent']
    pct_list.append(pct)
    
    try:
        pars, pcov = curve_fit(lin_decay, xdata=week, ydata=FVC, p0=[-0.5, 3000])
        pars_lin.append(pars)
        pcov_lin.append(pcov)
        RSS_lin += np.sum((FVC-lin_decay(week,*pars))**2)
        
        pars, pcov = curve_fit(exp_decay, xdata=week, ydata=FVC, p0=[-0.5, 500, 2000])
        pars_exp.append(pars)
        pcov_exp.append(pcov)
        RSS_exp += np.sum((FVC-exp_decay(week,*pars))**2)
        
        pars, pcov = curve_fit(log_decay, xdata=week, ydata=FVC, p0=[0.5, -0.5, 5000])
        pars_log.append(pars)
        pcov_log.append(pcov)
        RSS_log += np.sum((FVC-log_decay(week,*pars))**2)

        #fit_values.append([pid] + pars.tolist())
    except RuntimeError:
        pass
    
print("RSS for linear decay model: %.3f \n" % RSS_lin)   
print("RSS for exponential decay model: %.3f \n" % RSS_exp) 
print("RSS for logistic decay model: %.3f \n" % RSS_log)    
#     plt.plot(week,pct)
#     plt.xlabel('Weeks')
#     plt.ylabel('Percent')
#fit_raw = pd.DataFrame(fit_values, columns=columns).sort_values(by='r')
#fit_raw = fit_raw.reset_index(drop=True)

ax = fit_raw['r'].hist()
ax.set_xlabel('r')
ax.set_ylabel('counts')
ax.set_title("Initial fitting with individual patient's meta data")
ss_res = np.dot((pct - pct_model(week, *pars)),(pct - pct_model(week, *pars)))
ss_res
z=train_df.groupby(['SmokingStatus','Sex'])['FVC'].count().to_frame().reset_index()
z.columns = ['SmokingStatus', 'Sex', 'Count']
z.style.background_gradient(cmap='YlOrRd') 

imdir = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/'


fig=plt.figure(figsize=(13, 10))
columns = 5
rows = 3
interval = 2
imlist = os.listdir(imdir)
for i in range(1,columns*rows+1):
    loc = 1
    filename = imdir + "/" + str(i*interval) + ".dcm"
    ds = pydicom.dcmread(filename)
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap='YlOrRd')
    plt.yticks([])
    plt.xticks([])
    

# plt.tight_layout()
# plt.show()