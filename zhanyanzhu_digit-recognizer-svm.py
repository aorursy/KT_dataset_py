#load data
import pandas as pd
from sklearn.preprocessing import scale
train = pd.read_csv('../input/digit-recognizer/train.csv') 
test = pd.read_csv('../input/digit-recognizer/test.csv')
train.shape
test.shape
#parse data
y_data = train['label']
x_data = train.drop(columns = 'label')
#scale data
x_data = x_data/255.0
x_data = scale(x_data)
#split into training set and validation set in the ratio of 7:3
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 20)
#linear model
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
linear_model = SVC(kernel='linear')
linear_model.fit(x_train, y_train)
y_ = linear_model.predict(x_test)
print("linear model accuracy score", accuracy_score(y_test, y_))
#non linear model
nonlinear_model = SVC(kernel='rbf')
nonlinear_model.fit(x_train, y_train)
y_= nonlinear_model.predict(x_test)
print("non linear model accuracy score", accuracy_score(y_test, y_))
#grid serach (non linear model)
from sklearn.model_selection import GridSearchCV
tuned_parameters=  [ {'gamma': [1, 0.1, 0.01, 0.001], 'C': [1, 10, 100, 1000]}]
svc_rbf = SVC(kernel='rbf')
rbf_cv = GridSearchCV(estimator = svc_rbf, param_grid = tuned_parameters, cv = 5,return_train_score=True, n_jobs=-1) 
rbf_cv.fit(x_train, y_train)
result_rbf_c = pd.DataFrame(rbf_cv.cv_results_)
#test accuracy
svc_final = SVC(kernel='rbf', C=100, gamma = 0.001)
svc_final.fit(x_train, y_train)
y_predict_final = svc_final.predict(x_test)
print("test accuracy - non linear model", accuracy_score(y_test, y_predict_final))
svc_linear = SVC(kernel='linear')
params = {"C": [1, 10, 100, 1000]}
svc_grid = GridSearchCV(estimator = svc_linear, param_grid = params, cv=5, return_train_score=True)
svc_grid.fit(x_train, y_train)
result = pd.DataFrame(svc_grid.cv_results_)
#output result
test = test/255.0
test = scale(test)
svc_final = SVC(kernel='rbf', C=100, gamma = 0.001)
svc_final.fit(x_train, y_train)
result=svc_final.predict(test)
index = np.arange(1,28001,1)
import numpy as np
submission = pd.DataFrame({
    'ImageId':np.arange(1,28001,1),'Label':result
})
submission.index=submission['ImageId'].values
submission.to_csv('output.csv', index = False)