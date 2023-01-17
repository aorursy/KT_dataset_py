import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
import numpy as np
raw = pd.read_csv("../input/digit-recognizer/train.csv")
raw.shape # n: 42000, p: 785
raw["label"] = raw["label"].apply(str) # converting to factor values
def bi(num):
    if num > 0:
        return 1
    else:
        return 0
for i in raw.columns[1:]:
    raw.loc[:,i] = raw.loc[:,i].apply(bi)
for i in raw.nunique()[1:]:
    if i > 2:
        print(False)
        break
x = raw.drop(['label'], axis='columns')
y = raw["label"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#standardized data
ss = StandardScaler().fit(x_train)
x_train_scaled = ss.transform(x_train)
x_test_scaled = ss.transform(x_test) # scales the test x based on the scaling applied to x_train

sklearn_pca = sklearnPCA().fit(x_train_scaled)

per_var = sklearn_pca.explained_variance_ratio_
cum_per_var = sklearn_pca.explained_variance_ratio_.cumsum()
n_comp=len(cum_per_var[cum_per_var <= 0.90]) # Using number of PCs which explains 90% of the variance
n_comp
sklearn_pca = sklearnPCA(n_components=n_comp)
x_train_pca = sklearn_pca.fit_transform(x_train_scaled)
x_test_pca = sklearn_pca.transform(x_test_scaled)
c=[]
for i in range(-5, 17, 2):
    c.append(2**i)
start = time.time()
svc = SVC(kernel = "linear", decision_function_shape = "ovr")
parameters = {"C": c}
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(x_train_pca[:5000], y_train[:5000])
end = time.time()
print(f"time taken to search {end-start}")
clf.best_params_
clf.best_score_
clf = SVC(kernel = "linear", decision_function_shape = "ovr", C = 0.03125)
clf.fit(x_train_pca, y_train)
print(f"test accuracy: {clf.score(x_test_pca,y_test)}")
c=[]
for i in range(-5, 17, 2):
    c.append(2**i)
    
r=[]
for i in range(-15, 5, 2):
    r.append(2**i)
start = time.time()
svc = SVC(kernel = "rbf", decision_function_shape = "ovr")
parameters = {"C": c, "gamma": r}
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(x_train_pca[:5000], y_train[:5000])
end = time.time()
print(f"time taken to search {end-start}")
clf.best_params_
clf.best_score_
clf = SVC(kernel = "rbf", decision_function_shape = "ovr", C = 8, gamma = 0.001953125)
clf.fit(x_train_pca, y_train)
print(f"test accuracy: {clf.score(x_test_pca,y_test)}")
c=[]
for i in range(-5, 17, 2):
    c.append(2**i)
    
d=[2,3,4,5]
start = time.time()
svc = SVC(kernel = "poly", decision_function_shape = "ovr")
parameters = {"C": c, "degree": d}
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(x_train_pca[:5000], y_train[:5000])
end = time.time()
print(f"time taken to search {end-start}")
clf.best_params_
clf.best_score_
clf = SVC(kernel = "poly", decision_function_shape = "ovr", C = 128, degree = 3)
clf.fit(x_train_pca, y_train)
print(f"test accuracy: {clf.score(x_test_pca,y_test)}")
x_train = x
y_train = y
x_test = pd.read_csv("../input/digit-recognizer/test.csv")
for i in x_test.columns:
    x_test.loc[:,i] = x_test.loc[:,i].apply(bi)
#standardized data
ss = StandardScaler().fit(x_train)
x_train_scaled = ss.transform(x_train)
x_test_scaled = ss.transform(x_test) # scales the test x based on the scaling applied to x_train

sklearn_pca = sklearnPCA().fit(x_train_scaled)

per_var = sklearn_pca.explained_variance_ratio_
cum_per_var = sklearn_pca.explained_variance_ratio_.cumsum()
n_comp=len(cum_per_var[cum_per_var <= 0.90])
n_comp
sklearn_pca = sklearnPCA(n_components=n_comp)
x_train_pca = sklearn_pca.fit_transform(x_train_scaled)
x_test_pca = sklearn_pca.transform(x_test_scaled)
def kcv_poly_svm(i, cost): # i is a cluster, ie. range from 0-9
    start = i * 4200# length of df: 42000
    end = start + 4260
    
    test_x = x_train_pca[start:end]
    test_y = y_train[start:end]
    selected = list(range(start)) + list(range(end, 42000)) # exclude
    train_x = x_train_pca[selected]
    train_y = y_train.iloc[selected,]
    
    clf = SVC(kernel='poly', decision_function_shape = "ovr", C = cost, random_state = 1)
    clf.fit(train_x, train_y)
    
    return clf.score(test_x, test_y)# testset error
def cv_runner(c):
    cv_error_1 = []
    if __name__ == '__main__':
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()-1) as executor:
            for cost in c:
                class_error = []
                results = [executor.submit(kcv_poly_svm, i, cost) for i in range(10)]
                # results = executor.map(kcv_lienar_svm, range(10), 1)
                for r in concurrent.futures.as_completed(results):
                    class_error.append(r.result())
                cv_error_1.append(sum(class_error)/10)
                
    print(c)
    print(cv_error_1)
#c = [100, 200, 300, 400]
#cv_runner(c) # takes approximately 25 min to execute

#c = [80, 90, 100, 110, 120, 130, 140, 150]
#cv_runner(c)
clf = SVC(kernel = "poly", decision_function_shape = "ovr", C = 100, degree = 3, random_state=1)
clf.fit(x_train_pca, y_train)
prediction = clf.predict(x_test_pca)
out = pd.DataFrame({"ImageId": range(1,len(x_test)+1), "Label" : prediction})
out.to_csv("submission_final.csv", index = False)