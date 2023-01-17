%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals.six import StringIO
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pydotplus
df = pd.read_csv("../input/mushrooms.csv", dtype="category")
columns = df.columns.values

print("データ数：", len(df.index))
df.head()
df.isnull().sum()
# matplotlibで描画
for column in columns :
    plt.figure(figsize=(3,2))
    cat = df[column].value_counts().sort_index()
    plt.title(column)
    display(cat.index.codes)
    plt.bar(cat.index.codes, cat.values, tick_label=cat.index.tolist())
    plt.show()
# seabornで描画
for column in columns:
    ax = sns.factorplot(column, col="class", data=df, size=3.0, aspect=.9, kind="count")
df['class'] = df['class'].map({'e':0, 'p':1})
df['ring-number'] = df['ring-number'].map({'n':0, 'o':1, 't':2})
df = df.drop(['stalk-surface-below-ring','stalk-color-below-ring'], axis=1)
df_dummy = df.copy()
dummy_columns = ['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-color-above-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']

for column in dummy_columns:
    df_dummy[column+'_str'] = df[column].astype(str).map(lambda x:column+"_"+x)
    df_dummy = pd.concat([df_dummy, pd.get_dummies(df_dummy[column+'_str'])],axis=1)

drop_columns = dummy_columns.copy()
for column in dummy_columns:
    drop_columns.append(column+"_str")

df_dummy_fin = df_dummy.drop(drop_columns,axis=1)
df_dummy_fin.head()
df2 = df_dummy_fin.drop(['class'], 1)
df2.corr().style.background_gradient().format('{:.2f}')
# グローバル変数として定義
var2,var3,var4,var5,var6,var7,var8,var9,var10 = [],[],[],[],[],[],[],[],[]
# 相関係数のチェック用関数
def check_coef(df):

    # 初期化
    var10.clear(), var9.clear(), var8.clear(), var7.clear(), var6.clear(), var5.clear(), var4.clear(), var3.clear(), var2.clear()

    for v1 in df.columns:
        for v2 in df.columns:
            if v1==v2:
                continue
            else:
                coef = np.corrcoef(df[v1], df[v2])[0,1]
                cont = v1+","+str(df[v1][df[v1]==1].sum())+",  "+v2+","+str(df[v2][df[v2]==1].sum())
                if abs(coef) == 1.0:
                    var10.append(cont)
                elif abs(coef) >= 0.9 and abs(coef) < 1.0:
                    var9.append(cont)
                elif abs(coef) >= 0.8 and abs(coef) < 0.9:
                    var8.append(cont)
                elif abs(coef) >= 0.7 and abs(coef) < 0.8:
                    var7.append(cont)
                elif abs(coef) >= 0.6 and abs(coef) < 0.7:
                    var6.append(cont)
                elif abs(coef) >= 0.5 and abs(coef) < 0.6:
                    var5.append(cont)
                elif abs(coef) >= 0.4 and abs(coef) < 0.5:
                    var4.append(cont)
                elif abs(coef) >= 0.3 and abs(coef) < 0.4:
                    var3.append(cont)
                elif abs(coef) >= 0.2 and abs(coef) < 0.3:
                    var2.append(cont)
check_coef(df2)
sorted(set(var10))
drop_df_10 = ['bruises_t','gill-attachment_a','gill-spacing_w','ring-number_0','ring-type_n','stalk-color-above-ring_c','stalk-color-above-ring_y','gill-size_n','stalk-shape_e']
df_dummy_fin_10 = df_dummy_fin.drop(drop_df_10, 1)

check_coef(df_dummy_fin_10.drop(['class'],1))
sorted(set(var10))
x_train10, x_test10, y_train10, y_test10 = train_test_split(df_dummy_fin_10.drop("class",axis=1), df_dummy_fin_10['class'], test_size=0.3, random_state=12345)
lr = LogisticRegression()
lr.fit(x_train10,y_train10)
print(classification_report(y_test10,lr.predict(x_test10)))
print(confusion_matrix(y_test10,lr.predict(x_test10)))
check_coef(df_dummy_fin_10.drop(['class'], 1))
sorted(set(var9))
drop_df_9 = ['stalk-color-above-ring_o','veil-color_w','ring-number_2']
df_dummy_fin_9 = df_dummy_fin_10.drop(drop_df_9,axis=1)

check_coef(df_dummy_fin_9.drop(['class'],1))
sorted(set(var9))
x_train9, x_test9, y_train9, y_test9 = train_test_split(df_dummy_fin_9.drop("class",axis=1), df_dummy_fin_9['class'], test_size=0.3, random_state=12345)

lr = LogisticRegression()
lr.fit(x_train9,y_train9)
print(classification_report(y_test9,lr.predict(x_test9)))
print(confusion_matrix(y_test9,lr.predict(x_test9)))
check_coef(df_dummy_fin_9.drop(['class'], 1))
sorted(set(var8))
drop_df_8 = ['gill-color_b','ring-type_l','stalk-root_?','stalk-surface-above-ring_k']
df_dummy_fin_8 = df_dummy_fin_9.drop(drop_df_8,axis=1)

check_coef(df_dummy_fin_8.drop(['class'],1))
sorted(set(var8))
x_train8, x_test8, y_train8, y_test8 = train_test_split(df_dummy_fin_8.drop("class",axis=1), df_dummy_fin_8['class'], test_size=0.3, random_state=12345)

lr = LogisticRegression()
lr.fit(x_train8,y_train8)
print(classification_report(y_test8,lr.predict(x_test8)))
print(confusion_matrix(y_test8,lr.predict(x_test8)))
check_coef(df_dummy_fin_8.drop(['class'], 1))
sorted(set(var7))
drop_df_7 = ['ring-type_p','cap-shape_f','gill-color_e','population_c','stalk-color-above-ring_e','spore-print-color_h']
df_dummy_fin_7 = df_dummy_fin_8.drop(drop_df_7,axis=1)

check_coef(df_dummy_fin_7.drop(['class'],1))
sorted(set(var7))
x_train7, x_test7, y_train7, y_test7 = train_test_split(df_dummy_fin_7.drop("class",axis=1), df_dummy_fin_7['class'], test_size=0.3, random_state=12345)

lr = LogisticRegression()
lr.fit(x_train7,y_train7)
print(classification_report(y_test7,lr.predict(x_test7)))
print(confusion_matrix(y_test7,lr.predict(x_test7)))
check_coef(df_dummy_fin_7.drop(['class'], 1))
list(set(var6))
drop_df_6 = ['ring-type_e','veil-color_n','stalk-color-above-ring_p','veil-color_o','habitat_m','spore-print-color_w']
df_dummy_fin_6 = df_dummy_fin_7.drop(drop_df_6,1)

check_coef(df_dummy_fin_6.drop(['class'],1))
sorted(set(var6))
x_train6, x_test6, y_train6, y_test6 = train_test_split(df_dummy_fin_6.drop("class",axis=1), df_dummy_fin_6['class'], test_size=0.3, random_state=12345)

lr = LogisticRegression()
lr.fit(x_train6,y_train6)
print(classification_report(y_test6,lr.predict(x_test6)))
print(confusion_matrix(y_test6,lr.predict(x_test6)))
len(df_dummy_fin_6.columns)
check_coef(df_dummy_fin_6.drop(['class'], 1))
list(set(var5))
drop_df_5 = ['stalk-root_e','stalk-root_c','odor_a','odor_l','population_n','population_a','habitat_w','gill-color_y','spore-print-color_r','gill-color_o','ring-type_f','gill-color_r','stalk-surface-above-ring_y','cap-color_r','cap-color_u','veil-color_y','cap-surface_f','cap-surface_s','habitat_g','population_y','odor_f','population_v','habitat_d','bruises_f']
df_dummy_fin_5 = df_dummy_fin_6.drop(drop_df_5,1)

check_coef(df_dummy_fin_5.drop(['class'],1))
sorted(set(var5))
x_train5, x_test5, y_train5, y_test5 = train_test_split(df_dummy_fin_5.drop("class",axis=1), df_dummy_fin_5['class'], test_size=0.3, random_state=12345)

lr = LogisticRegression()
lr.fit(x_train5,y_train5)
print(classification_report(y_test5,lr.predict(x_test5)))
print(confusion_matrix(y_test5,lr.predict(x_test5)))
check_coef(df_dummy_fin_5.drop(['class'], 1))
list(set(var4))
drop_df_4 = ['habitat_l','odor_s','odor_y','stalk-surface-above-ring_f','spore-print-color_b','spore-print-color_o','spore-print-color_y','cap-color_y','odor_n','stalk-root_b']
df_dummy_fin_4 = df_dummy_fin_5.drop(drop_df_4,1)

check_coef(df_dummy_fin_4.drop(['class'],1))
sorted(set(var4))
x_train4, x_test4, y_train4, y_test4 = train_test_split(df_dummy_fin_4.drop("class",axis=1), df_dummy_fin_4['class'], test_size=0.3, random_state=12345)

lr = LogisticRegression()
lr.fit(x_train4,y_train4)
print(classification_report(y_test4,lr.predict(x_test4)))
print(confusion_matrix(y_test4,lr.predict(x_test4)))
check_coef(df_dummy_fin_4.drop(['class'], 1))
list(set(var3))
drop_df_3 = ['cap-color_g','cap-color_e','population_s','cap-color_w','cap-shape_k','gill-color_g','stalk-color-above-ring_g','stalk-color-above-ring_n','stalk-color-above-ring_b','habitat_u','odor_p','odor_c','cap-color_p','spore-print-color_k','spore-print-color_n','cap-surface_y','stalk-color-above-ring_w','stalk-shape_t']
df_dummy_fin_3 = df_dummy_fin_4.drop(drop_df_3,1)

check_coef(df_dummy_fin_3.drop(['class'],1))
sorted(set(var3))
x_train3, x_test3, y_train3, y_test3 = train_test_split(df_dummy_fin_3.drop("class",axis=1), df_dummy_fin_3['class'], test_size=0.3, random_state=12345)

lr = LogisticRegression()
lr.fit(x_train3,y_train3)
print(classification_report(y_test3,lr.predict(x_test3)))
print(confusion_matrix(y_test3,lr.predict(x_test3)))
len(df_dummy_fin_3.columns)
check_coef(df_dummy_fin_3.drop(['class'], 1))
list(set(var2))
drop_df_2 = ['habitat_p','gill-color_n','gill-color_h','gill-color_w','gill-spacing_c','cap-shape_b','cap-color_c','cap-color_n','odor_m','cap-surface_g','cap-shape_c']
df_dummy_fin_2 = df_dummy_fin_3.drop(drop_df_2,1)

check_coef(df_dummy_fin_2.drop(['class'],1))
sorted(set(var2))
x_train2, x_test2, y_train2, y_test2 = train_test_split(df_dummy_fin_2.drop("class",axis=1), df_dummy_fin_2['class'], test_size=0.3, random_state=12345)

lr = LogisticRegression()
lr.fit(x_train2,y_train2)
print(classification_report(y_test2,lr.predict(x_test2)))
print(confusion_matrix(y_test2,lr.predict(x_test2)))
len(df_dummy_fin_2.columns)
#df_data = df_dummy_fin_2
df_data = df_dummy_fin_3
train = df_data[:6000]
test = df_data[6000:-1]

y_train = train['class']
y_test = test['class']
x_train = train.drop('class', 1)
x_test = test.drop('class', 1)
y_train.plot(kind='hist', figsize=(3,2))
y_test.plot(kind='hist', figsize=(3,2))
x_train, x_test, y_train, y_test = train_test_split(df_data.drop("class",axis=1), df_data['class'], test_size=0.3, random_state=12345)
y_train.plot(kind='hist', figsize=(3,2))
y_test.plot(kind='hist', figsize=(3,2))
print("トレーニングデータのベースレート：", "{:.1%}".format(y_train[y_train==1].count() / y_train.count()))
print("テストデータのベースレート：", "{:.1%}".format(y_test[y_test==1].count() / y_test.count()))
# ロジスティック回帰モデルを使って学習する
clf_lr = LogisticRegression()
clf_lr.fit(x_train,y_train)
#print(clf_lr.coef_,clf_lr.intercept_)
print(classification_report(y_test,clf_lr.predict(x_test)))
print(confusion_matrix(y_test,clf_lr.predict(x_test)))
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
X, y = x_train.values.astype(np.float64), y_train.values.astype(np.float64)

for nn in range(2,10):
    print("n_neighbors=%s"%nn)
    
    knn = KNeighborsClassifier(n_neighbors=nn)
    kfold = StratifiedKFold(n_splits=5,random_state=1234).split(X, y)

    scores = []
    for k, (train_index, test_index) in enumerate(kfold):
        X_train_knn, X_test_knn = X[train_index], X[test_index]
        y_train_knn, y_test_knn = y[train_index], y[test_index]

        # 標準化
        stdsc = StandardScaler()
        X_train_std = stdsc.fit_transform(X_train_knn)
        X_test_std = stdsc.transform(X_test_knn)

        knn.fit(X_train_std, y_train_knn)
        score = knn.score(X_train_std, y_train_knn)
        scores.append(score)
        print('Fold: %s, Class dist: %s, Accuracy: %.3f' % (k+1, np.bincount(y_train_knn.astype(np.int64)), score))

    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    print()
# 標準化
stdsc = StandardScaler()
x_train_std = stdsc.fit_transform(x_train)
x_test_std = stdsc.fit_transform(x_test)

param_grid = {'n_neighbors':range(2,10)}
clf_knn = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=5)
clf_knn.fit(x_train_std, y_train)
print(classification_report(y_test, clf_knn.best_estimator_.predict(x_test)))
print(confusion_matrix(y_test, clf_knn.best_estimator_.predict(x_test)))
from sklearn.tree import DecisionTreeClassifier, export_graphviz
param_grid = {'criterion':['gini','entropy'], 'max_depth':range(2,10), 'min_samples_split':range(2,5), 'min_samples_leaf':range(2,5)}
clf_tree = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=5)
clf_tree.fit(x_train,y_train)
clf_tree.best_estimator_
clf_tree.best_params_
print(clf_tree.best_estimator_.feature_importances_)
pd.DataFrame(clf_tree.best_estimator_.feature_importances_, index=df_data.columns.values[1:]).plot.bar(figsize=(20,5))
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()
dot_data = StringIO() #dotファイル情報の格納先
export_graphviz(clf_tree.best_estimator_, out_file=dot_data,  
                     feature_names=df_data.columns.values[0:-1],  
                     class_names=["0","1"],  
                     filled=True, rounded=True,  
                     special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())
y_pred_tree = clf_tree.predict(x_test)
print(classification_report(y_test,y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
from sklearn.ensemble import RandomForestClassifier
param_grid = {'n_estimators':range(2,13), 'max_depth':range(2,10), 'min_samples_split':range(2,5), 'min_samples_leaf':range(2,5)}
clf_rf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5)
clf_rf.fit(x_train, y_train)
clf_rf.best_estimator_
clf_rf.best_params_
print(clf_rf.best_estimator_.feature_importances_)
pd.DataFrame(clf_rf.best_estimator_.feature_importances_, index=df_data.columns.values[1:]).plot.bar(figsize=(20,5))
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()
y_pred_rf = clf_rf.best_estimator_.predict(x_test)
print(classification_report(y_test,y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
from sklearn.ensemble import AdaBoostClassifier
param_grid = {'n_estimators':range(2,15), 'learning_rate':[x*0.1 for x in range(1,10)], 'algorithm':['SAMME','SAMME.R']}
clf_ada = GridSearchCV(AdaBoostClassifier(), param_grid=param_grid, cv=5)
clf_ada.fit(x_train, y_train)
clf_ada.best_estimator_
clf_ada.best_params_
y_pred_ada = clf_ada.best_estimator_.predict(x_test)
print(classification_report(y_test,y_pred_ada))
print(confusion_matrix(y_test, y_pred_ada))
from sklearn.svm import SVC
%%time
parameters = {'kernel':['linear', 'rbf'], 'C':[1, 5], 'gamma':[0.1*x for x in range(1,10)]}
model = SVC(probability=True)
clf_svm = GridSearchCV(model, parameters, cv=5)
clf_svm.fit(x_train, y_train)
print(clf_svm.best_params_, clf_svm.best_score_)
y_pred_svm = clf_svm.best_estimator_.predict(x_test)
print(classification_report(y_test,y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print("ロジスティック回帰")
print()
print(classification_report(y_test,clf_lr.predict(x_test)))

print("KNN")
print()
print(classification_report(y_test, clf_knn.best_estimator_.predict(x_test)))

print("決定木")
print()
print(classification_report(y_test, clf_tree.best_estimator_.predict(x_test)))

print("ランダムフォレスト")
print()
print(classification_report(y_test, clf_rf.best_estimator_.predict(x_test)))

print("アダブースト")
print()
print(classification_report(y_test, clf_ada.best_estimator_.predict(x_test)))

print("SVM")
print()
print(classification_report(y_test, clf_svm.best_estimator_.predict(x_test)))
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1)

# まずはランダム線を引く
ax.plot([0, 1], [0, 1], color="gray", alpha=0.2, linestyle="--")

# 各モデルのROC曲線を引く
clf_all = [clf_lr, clf_knn, clf_tree, clf_rf, clf_ada, clf_svm]
clf_labels = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "Ada Boost", "SVM"]
colors = ["red", "blue", "green", "yellow", "orange", "pink"]
linestyles = [":", "--", "-.", "-", ":", "-."]

for clf, label, clr, ls in zip(clf_all, clf_labels, colors, linestyles):
    roc_auc = roc_auc_score(y_test, clf.predict(x_test))
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(x_test)[:, 1])
    ax.step(fpr, tpr, color=clr, linestyle=ls, label="%s (AUC = %0.2f)" % (label, roc_auc))

plt.legend()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title("ROC Curve")
plt.xlabel("FP: False Positive Rate")
plt.ylabel("TP: True Positive Rate")
plt.grid(True)
df_y = pd.DataFrame()
df_y['class'] = df_data['class']

df_x = pd.DataFrame()
df_x['gill-spacing_c'] = df_data['gill-spacing_c']
df_x['gill-size_b'] = df_data['gill-size_b']
df_x['stalk-surface-above-ring_s'] = df_data['stalk-surface-above-ring_s']
x_train_t1, x_test_t1, y_train_t1, y_test_t1 = train_test_split(df_x, df_y, test_size=0.3, random_state=12345)
param_grid = {'criterion':['gini','entropy'], 'max_depth':range(2,10), 'min_samples_split':range(2,5), 'min_samples_leaf':range(2,5)}
clf_tree2 = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=5)
clf_tree2.fit(x_train_t1, y_train_t1)
print(classification_report(y_test_t1,clf_tree2.predict(x_test_t1)))
print(confusion_matrix(y_test_t1,clf_tree2.predict(x_test_t1)))
