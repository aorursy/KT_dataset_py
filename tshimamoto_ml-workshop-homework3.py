%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
############################# 
# 欠測値の変換
############################## 
def replace_nan(df, c, replace):
    df.loc[df[c].isnull(), c] = replace
    return df

############################## 
# one-hot encodingの実行
############################## 
def create_dummy_var(orgdf, orgcol):
    tempcol = orgcol + '_str'
    # 一時列を作る。値は＜元の列名＞_＜カテゴリ値＞
    orgdf[tempcol] = orgdf[orgcol].astype(str).map(lambda x : orgcol + '_' + x)
    # 一時列の値をダミー変数として追加する。
    newdf = pd.concat([orgdf, pd.get_dummies(orgdf[tempcol])], axis=1)
    # 一時列と元の列は削除する。
    newdf.drop([tempcol, orgcol], axis=1, inplace=True)
    return newdf

############################## 
# 前処理全般
#     ・欠測値の処理
#     ・異常値の処理
#     ・特徴量エンジニアリング
#     ・one-hotencodingの実行
#     ・マルチコの検出
############################## 
def preprocess():
    df = pd.read_csv('../input/survey.csv')

    ############################## 
    # 欠測値の処理
    ############################## 
    # 各変数について欠測値処理。
    df = replace_nan(df, 'state', 'None')
    df = replace_nan(df, 'self_employed', 'No')
    df = replace_nan(df, 'work_interfere', 'Never')

    # 変数commentsについての欠測値処理。コメントの内容に関係なく、欠測値は一律0（コメント無し）とした。
    df = replace_nan(df, 'comments', 0)

    ############################## 
    # 異常値の処理
    ############################## 
    # Age:負の値、100を超過する値は異常値とし、データ行削除。
    c = 'Age'
    df.drop(df.index[(df[c] < 0) | (100 < df[c])], inplace=True)

    # Gender:男性をM、女性をF、両性をHに名寄せ。解釈不明な入力値はNaNにしてデータ行削除。
    c = 'Gender'
    replace_map = {'Female':'F','M':'M','Male':'M','male':'M','female':'F','m':'M','Male-ish':'M','maile':'M','Trans-female':'F','Cis Female':'F','F':'F','something kinda male?':np.nan,'Cis Male':'M','Woman':'F','f':'F','Mal':'M','Male (CIS)':'M','queer/she/they':'H','non-binary':'H','Femake':'F','woman':'F','Make':'M','Nah':np.nan,'All':'H','Enby':'H','fluid':'H','Genderqueer':'H','Female ':'F','Androgyne':'H','Agender':'H','cis-female/femme':'F','Guy (-ish) ^_^':'M','male leaning androgynous':'H','Male ':'M','Man':'M','Trans woman':'F','msle':'M','Neuter':'H','Female (trans)':'F','queer':'H','Female (cis)':'F','Mail':'M','cis male':'M','A little about you':np.nan,'Malr':'M','p':np.nan,'femail':'F','Cis Man':'M','ostensibly male, unsure what that really means':np.nan}
    df[c].replace(replace_map, inplace=True)
    df.drop(df.index[df[c].isnull()], inplace=True)

    ############################## 
    # 特徴量エンジニアリング
    ############################## 
    # コメント欄：空でなければ1（コメントあり）、空ならば0に置換。
    # なお、Notebookのセルを連続して実行してもエラーならないように２重実行を防止しておく。
    # ※一度実行すると変数の型が自動的にbool型に変わることを利用して、元のobject型であるときだけ実行するように制御。
    c = 'comments'
    df.loc[(df[c]!=0)&(df[c].str.strip()!=''), c] = 1

    # 01変換（対象は2値変数）
    replace_map = {'Yes':1, 'No':0}
    # なお、Notebookのセルを連続して実行してもエラーならないように２重実行を防止しておく。
    # ※一度実行すると変数の型が自動的にint64型に変わることを利用して、元のobject型であるときだけ実行するように制御。
    # print(df.dtypes['treatment'])
    if (df.dtypes['treatment'] == 'object'):
        df['treatment'].replace(replace_map, inplace=True)
        df['family_history'].replace(replace_map, inplace=True)
        df['obs_consequence'].replace(replace_map, inplace=True)

    # 説明変数としなかった変数は削除。
    df.drop([
        'Timestamp','Age','Country','self_employed','no_employees','remote_work',
        'tech_company','wellness_program','seek_help','mental_health_consequence','phys_health_consequence','coworkers',
        'supervisor','mental_health_interview','phys_health_interview','mental_vs_physical'
        ], axis=1, inplace=True)

    ############################## 
    # onehot-encodingの準備
    ############################## 
    # ただし、onehot-encodingの前に、新規変数名を"＜元の変数名＞_＜カテゴリ値＞"とするためにスペースやシングルコーテーションを含むカテゴリ値を適切に変換しておく。
    # 変数benefits、anonymity
    replace_map={"Don't know":'DontKnow'}
    df['benefits'].replace(replace_map, inplace=True)
    df['anonymity'].replace(replace_map, inplace=True)

    # 変数care_options
    replace_map={"Not sure":'NotSure'}
    df['care_options'].replace(replace_map, inplace=True)

    # 変数leave
    replace_map={'Very easy':'VeryEasy', 'Somewhat easy':'SomewhatEasy', 'Somewhat difficult':'SomewhatDifficult', 'Very difficult':'VeryDifficult', "Don't know":'DontKnow'}
    df['leave'].replace(replace_map, inplace=True)

    ############################## 
    # onehot-encoding実行
    ############################## 
    df_fin = df # 最初は元のdfを引数としているので注意。
    df_fin = create_dummy_var(df_fin, 'Gender')
    df_fin = create_dummy_var(df_fin, 'state')
    df_fin = create_dummy_var(df_fin, 'work_interfere')
    df_fin = create_dummy_var(df_fin, 'benefits')
    df_fin = create_dummy_var(df_fin, 'care_options')
    df_fin = create_dummy_var(df_fin, 'anonymity')
    df_fin = create_dummy_var(df_fin, 'leave')

    # 後々の見易さのために目的変数treatmentを一番左に移動しておく。
    cols = df_fin.columns.tolist()
    cols.remove('treatment')
    cols.insert(0, 'treatment')
    df_fin = df_fin[cols]
    #df_fin.head()

    ############################## 
    # マルチコの検出
    ############################## 
    corr = df_fin.corr().style.background_gradient().format('{:.2f}')
    #display(corr)

    # 相関が高い説明変数を削除。相関が高い説明変数の条件として、相関係数が0.5以上または-0.5以下とした。
    # 変数1|変数2|相関係数
    # ---|---|---
    # Gender_F|Gender_M|-0.98
    # work_interfere_Never|work_interfere_Sometimes|-0.60
    # benefits_DontKnow|benefits_Yes|-0.54
    # care_options_No|care_options_Yes|-0.60
    # anonymity_DontKnow|anonymity_Yes|-0.89
    df_fin.drop(['Gender_F','work_interfere_Never','benefits_DontKnow','care_options_No','anonymity_DontKnow'], axis=1, inplace=True)
    
    display(df_fin.head())
    return df_fin

##############################
# モデル生成・あてはめ実行メソッド
# 与えられたアルゴリズムとデータで学習（モデル作成）と予測（あてはめ）を行い、混同行列を表示するところまで行う。
##############################
def go_fit_pred(algo, df):
    # 説明変数をX、目的変数をyとしてロジスティック回帰を実行。
    X = df.drop(['treatment'], axis=1)
    y = df['treatment']
    algo.fit(X, y)
    # print(algo.coef_)
    # print(algo.intercept_)

    # 混同行列
    # 真の値の抽出
    y_true = df['treatment'].values
    # 予測値の算出。ただし、テストデータではなく元のデータセットを使った算出。
    y_pred = algo.predict(X)
    # なお、デフォルトの混同行列では行、列とも表示順が[0(陰性), 1(陽性)]となって理解しづらいので、行、列とも[1(陽性),0(陰性)]の順に変更。
    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[1, 0])
    print('行、列ともに[1(陽性),0(陰性)]の順に直した混同行列:\r\n{}'.format(confmat))

    # 分類レポート
    print('')
    print('分類レポート:')
    report = classification_report(y_true, y_pred, digits=3) # 小数点以下3桁まで表示
    print(report)
    
    # データが反映されたモデルを返す
    return algo

############################## 
# グリッドサーチ実行メソッド
##############################
def go_grid(estimator, param_grid, cv, X_train, X_test, y_train, y_test):
    gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, n_jobs=4) # 処理時間短縮のため並列度を指定。
    gs.fit(X_train, y_train)
    print('Best Params  : {0}'.format(gs.best_params_)) # 最良モデルのパラメータ
    print('Best Score   : %.3f' % gs.best_score_) # 最良モデルの平均スコア
    print('Test Accuracy: %.3f' % gs.best_estimator_.score(X_test, y_test)) # 最良モデルによるテストデータの正解率=(TP+TN)/(TP+TN+FP+FN)
    print('Classification Report:') # 最良モデルによるテストデータの分類レポート
    y_pred = gs.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3) # 小数点以下3桁まで表示
    print(report)

############################## 
# グリッドサーチ実行メソッド（scoring='precision'指定版）
##############################
def go_grid_with_precision(estimator, param_grid, cv, X_train, X_test, y_train, y_test):
    gs = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring='precision', n_jobs=4) # 処理時間短縮のため並列度を指定。
    gs.fit(X_train, y_train)
    print('Best Params  : {0}'.format(gs.best_params_))
    print('Best Score   : %.3f' % gs.best_score_)
    print('Test Accuracy: %.3f' % gs.best_estimator_.score(X_test, y_test))
    print('Classification Report:')
    y_pred = gs.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3) # 小数点以下3桁まで表示
    print(report)

############################## 
# AdaBoostメソッド
##############################
from sklearn.ensemble import AdaBoostClassifier
def go_adab(estimator, n_estimators, X_train, X_test, y_train, y_test):
    adab = AdaBoostClassifier(base_estimator=estimator, n_estimators=n_estimators, random_state=1234)
    adab.fit(X_train, y_train)
    print('Test Accuracy: %.3f' % adab.score(X_test, y_test))
    print('Classification Report:')
    y_pred = adab.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3) # 小数点以下3桁まで表示
    print(report)
df_fin = preprocess()
from sklearn.model_selection import train_test_split
X, y = df_fin.drop(['treatment'], axis=1), df_fin['treatment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
from sklearn.svm import SVC

print("########## linear ##########")
clf = SVC(C=10, kernel="linear")
clf.fit(X, y) 
clf = go_fit_pred(clf, df_fin)

print("########## Gaussian ##########")
clf = SVC(C=10, kernel="rbf")
clf.fit(X, y) 
clf = go_fit_pred(clf, df_fin)
%%time
param_grid={
    'kernel':['linear', 'rbf'],
    'C':np.arange(1, 11),
    'gamma':np.arange(1, 11)
}
go_grid(SVC(), param_grid, 5, X_train, X_test, y_train, y_test)
%%time
param_grid={
    'kernel':['linear', 'rbf'],
    'C':np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
    'gamma':np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0])
}
go_grid(SVC(), param_grid, 5, X_train, X_test, y_train, y_test)
%%time
# param_grid={
#     'kernel':['linear', 'rbf'],
#     'C':np.arange(1.0, 5.1, 0.1),
#     'gamma':np.arange(0.0, 0.101, 0.001)
# }
# 上記結果。時間がかかりすぎる。
# Best Params  : {'C': 1.1, 'gamma': 0.02, 'kernel': 'rbf'}
# Best Score   : 0.829
# Test Accuracy: 0.856
# Classification Report:
#              precision    recall  f1-score   support
#           0      0.937     0.749     0.832       179
#           1      0.806     0.954     0.874       196
# avg / total      0.869     0.856     0.854       375
# Wall time: 11min 51s

# Kaggle上での処理時間短縮のため、ローカルでいろいろ試した結果で範囲を絞った。
param_grid={
    'kernel':['rbf'],
    'C':np.arange(1.00, 1.21, 0.01),
    'gamma':np.arange(0.00, 0.03, 0.01)
}
go_grid(SVC(), param_grid, 5, X_train, X_test, y_train, y_test)
%%time
# param_grid={
#     'kernel':['linear', 'rbf'],
#     'C':np.arange(0.50, 1.51, 0.01),
#     'gamma':np.arange(0.00, 0.11, 0.01)
# }
# 上記結果。時間がかかりすぎる。
# Best Params  : {'C': 0.54, 'gamma': 0.01, 'kernel': 'rbf'}
# Best Score   : 0.778
# Test Accuracy: 0.800
# Classification Report:
#              precision    recall  f1-score   support
#           0      0.799     0.777     0.788       179
#           1      0.801     0.821     0.811       196
# avg / total      0.800     0.800     0.800       375
# Wall time: 3min 16s

# Kaggle上での処理時間短縮のため、ローカルでいろいろ試した結果で範囲を絞った。
param_grid={
    'kernel':['rbf'],
    'C':np.arange(0.50, 0.76, 0.01),
    'gamma':np.arange(0.00, 0.06, 0.01)
}
go_grid(SVC(), param_grid, 5, X_train, X_test, y_train, y_test)
# 上記一覧表のうち、(2)、(4-1)、(4-2)をグラフ化
result_dict = {
    "Algorithm":[
        "LogisticRegression", "LogisticRegression", "LogisticRegression", 
        "DecisionTree", "DecisionTree", "DecisionTree", 
        "RandomForest", "RandomForest", "RandomForest", 
        "KNeighbors", "KNeighbors", "KNeighbors", 
        "SVM", "SVM", "SVM" 
    ],
    "TryName":[
        "(2)", "(4-1)", "(4-2)",
        "(2)", "(4-1)", "(4-2)",
        "(2)", "(4-1)", "(4-2)",
        "(2)", "(4-1)", "(4-2)",
        "(2)", "(4-1)", "(4-2)"
    ],
    "Accuracy":[
        0.853, 0.853, 0.811, 0.829, 0.811, 0.781, 0.816, 0.789, 0.829, 0.725, 0.733, np.nan, 0.856, 0.856, np.nan
    ],
    "Precision":[
        0.837, 0.837, 0.824, 0.827, 0.834, 0.817, 0.804, 0.772, 0.833, 0.769, 0.864, np.nan, 0.806, 0.809, np.nan
    ],
    "Recall":[
        0.893, 0.893, 0.811, 0.852, 0.796, 0.750, 0.857, 0.847, 0.842, 0.679, 0.582, np.nan, 0.954, 0.949, np.nan 
    ],
    "F1":[
        0.864, 0.864, 0.817, 0.839, 0.815, 0.782, 0.830, 0.808, 0.838, 0.721, 0.695, np.nan, 0.874, 0.873, np.nan
    ]
}
result_df = pd.DataFrame.from_dict(result_dict)
result_df = replace_nan(result_df, "Accuracy", 0.0)
result_df = replace_nan(result_df, "Precision", 0.0)
result_df = replace_nan(result_df, "Recall", 0.0)
result_df = replace_nan(result_df, "F1", 0.0)
#display(result_df)

plt.style.use("ggplot")
fig, ax = plt.subplots(1, len(result_df.Algorithm.unique()), figsize=(20,4))

w = 0.1
algos = result_df.Algorithm.unique()
for i, algo in enumerate(algos):
    temp_df = result_df.loc[(result_df.Algorithm == algo)]
    wt = np.array([-0.2, 0.3, 0.8])
    ax[i].bar(wt, temp_df.Accuracy, label=temp_df.Accuracy.name, width=w, color="silver")
    wt = wt + w
    ax[i].bar(wt, temp_df.Precision, label=temp_df.Precision.name, width=w, color="red")
    wt = wt + w
    ax[i].bar(wt, temp_df.Recall, label=temp_df.Recall.name, width=w, color="lightyellow")
    wt = wt + w
    ax[i].bar(wt, temp_df.F1, label=temp_df.F1.name, width=w, color="lightblue")
    
    ax[i].set_xlim(-0.4, 1.2, 1)
    ax[i].set_xticks([-0.5, -0.1, 0.4, 0.9])
    ax[i].set_xticklabels(["", "(2)", "(4-1)", "(4-2)"])
    ax[i].set_xlabel(algo, fontsize=20)
    ax[i].set_ylim(0.55, 1.01, 0.01)
    if (i == 0):
        ax[i].set_ylabel("Rate", fontsize=20)
    else:
        ax[i].set_yticklabels([])
        ax[i].set_ylabel("")
    ax[i].minorticks_on()
    ax[i].grid(which="major", axis="y", color="white", linestyle="-", linewidth="1")
    ax[i].grid(which="minor", axis="y", color="lightgray", linestyle="--", linewidth="1")
    ax[i].tick_params(axis="x", which="major", labelsize=15)
    ax[i].legend()

fig.subplots_adjust(wspace=0.02)
plt.show()

from sklearn.tree import DecisionTreeClassifier, export_graphviz
clf = DecisionTreeClassifier(criterion="entropy", max_depth=7, min_samples_split=5, min_samples_leaf=2, random_state=1234)
clf = go_fit_pred(clf, df_fin)

# 重要度の表示
print(clf.feature_importances_)
columns=df_fin.drop('treatment', axis=1).columns
pd.DataFrame(clf.feature_importances_, index=columns).plot.bar(figsize=(20,5))
plt.ylabel("Importance", fontsize=20)
plt.xlabel("Features", fontsize=20)
plt.tick_params(axis="x", which="major", labelsize=15)
plt.show()
