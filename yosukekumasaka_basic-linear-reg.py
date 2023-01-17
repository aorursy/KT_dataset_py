# 下処理。
# PassengerId – 乗客識別ユニークID
# Survived – 生存フラグ（0=死亡、1=生存）
# Pclass – チケットクラス
# Name – 乗客の名前
# Sex – 性別（male=男性、female＝女性）
# Age – 年齢
# SibSp – タイタニックに同乗している兄弟/配偶者の数
# parch – タイタニックに同乗している親/子供の数
# ticket – チケット番号
# fare – 料金
# cabin – 客室番号
# Embarked – 出港地（タイタニックへ乗った港）
# 0:  PassengerId → ただのシーケンス番号。生存に無関係のはずなので無視。すべて0にする。
# 1:  Pclass → そのまま使う。
# 2:  Name → Mr/Mrs/Miss/MasterのOne-Hotにする
# 3:  Sex → Nameを使うので削除
# 4:  Age → 欠損値を埋める（便利関数を探す）
# 5:  SibSp → そのまま。
# 6:  Parch → そのまま。
# 7:  Ticket → とりあえず無視。すべて0にする。
# 8:  Fare → そのまま使う
# 9:  Cabin → とりあえず先頭1文字を使う。欠損値はZに。それをLabel Encoding
# 10:  Embarked → 欠損値を補い、Label Encoding
# ライブラリのインポート
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression

# Jupyter Notebookの中でインライン表示する場合の設定（これが無いと別ウィンドウでグラフが開く）
%matplotlib inline
# トレーニングデータとテストデータの読み込み。
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# トレーニングデータのProfile Reportを作成 （出力結果が膨大なのでコメントアウト。必要な時だけ実行）
# pandas_profiling.ProfileReport(train)
# シャッフル（trainとcvを分けるなら有効化）
# train = train.sample(frac=1) 

# trainの列構成の調整
X_train = train.drop(["PassengerId", "Survived"], axis=1)
y_train = train["Survived"]

# testの列構成の調整
X_test = test.drop(["PassengerId"], axis=1)
# データ下処理の関数の定義
def preprocess(df):
    # NameからMr/Mrs/Miss/Masterを取り出し
    df["Mr"] = df["Name"].apply(lambda x: x.count("Mr."))
    df["Mrs"] = df["Name"].apply(lambda x: x.count("Mrs."))
    df["Miss"] = df["Name"].apply(lambda x: x.count("Miss."))
    df["Master"] = df["Name"].apply(lambda x: x.count("Master."))

    # Name/SexにMr/Mrs/Miss/Master以外に欲しい情報は残っていないため削除
    df = df.drop("Name", axis=1)
    df = df.drop("Sex", axis=1)

    # Embarked (S/C/Q)
    embarked_dummies = pd.get_dummies(df['Embarked'])
    df = pd.concat([df.drop(['Embarked'],axis=1),embarked_dummies],axis=1)

    # Age、Fareの欠損値を埋める
    df = df.fillna({"Age": int(df.mean()["Age"])})
    df = df.fillna({"Fare": int(df.mean()["Fare"])})

    # Ticketの番号部分の特定範囲だけ、カテゴリとして抽出（Excelでの分析で、有意な偏りがあったため）
    def getTicketNoCategory(val):
        if (val is not None):
            s = val.rsplit(" ", 1)
            if 1 < len(s):
                return int(int(s[1]) / 100)
        return -1    
    df["TicketNo"] = df["Ticket"].apply(lambda x: getTicketNoCategory(x))
    df["T23"] = df["TicketNo"].apply(lambda x: 1 if x == 23 else 0)
    df["T118"] = df["TicketNo"].apply(lambda x: 1 if x == 118 else 0)
    df["T135"] = df["TicketNo"].apply(lambda x: 1 if x == 135 else 0)
    df["T175"] = df["TicketNo"].apply(lambda x: 1 if x == 175 else 0)
    df["T176"] = df["TicketNo"].apply(lambda x: 1 if x == 176 else 0)
    df["T178"] = df["TicketNo"].apply(lambda x: 1 if x == 178 else 0)
    df["T1138"] = df["TicketNo"].apply(lambda x: 1 if x == 1138 else 0)
    df["T2487"] = df["TicketNo"].apply(lambda x: 1 if x == 2487 else 0)
    df["T3151"] = df["TicketNo"].apply(lambda x: 1 if x == 3151 else 0)
    df["T3458"] = df["TicketNo"].apply(lambda x: 1 if x == 3458 else 0)
    df["T3471"] = df["TicketNo"].apply(lambda x: 1 if x == 3471 else 0)
    df["T3492"] = df["TicketNo"].apply(lambda x: 1 if x == 3492 else 0)
    df["T3500"] = df["TicketNo"].apply(lambda x: 1 if x == 3500 else 0)
    df["T31013"] = df["TicketNo"].apply(lambda x: 1 if x == 31013 else 0)
    df = df.drop(["Ticket", "TicketNo"], axis=1)
    
    
    # Cabin （1文字目の値をOne-Hot）
    df["Cabin1"] = df["Cabin"].apply(lambda x: "Cabin" + x[0] if type(x) is str else "")
    cabin1_dummies = pd.get_dummies(df['Cabin1'])
    df = pd.concat([df.drop(["Cabin", "Cabin1"],axis=1),cabin1_dummies],axis=1)
    df = df.drop([""],axis=1)
#    # Cabinの有無 → この入れ方だと下がった。他の説明変数との相関が原因かも。
#    df["CabinExist"] = df["Cabin"].apply(lambda x: 1 if type(x) is str else 0)
#    # 時間の都合上、Cabinは削除
#    df = df.drop(["Cabin"], axis=1)
    
    return df
# X_train_preprocessed = preprocess(X_train)
# print(np.std(X_train_preprocessed["T178"]))
# X_train_preprocessed = X_train_preprocessed.apply(lambda x: (x - np.mean(x)) / (np.std(x)) if np.std(x) != 0 else x)
# X_train_preprocessed
# 前処理の実行（train）
X_train_preprocessed = preprocess(X_train)
X_train_preprocessed = X_train_preprocessed.apply(lambda x: (x - np.mean(x)) / (np.std(x)) if np.std(x) != 0 else x)

# 前処理の実行（test）
X_test_preprocessed = preprocess(X_test)
X_test_preprocessed.isnull().any()
X_test_preprocessed = X_test_preprocessed.apply(lambda x: (x - np.mean(x)) / (np.std(x)) if np.std(x) != 0 else x)

# One-hotで絡む構成が会わない部分の調整（両方が持つカラム以外をドロップする）
X_train_preprocessed,X_test_preprocessed = X_train_preprocessed.align(X_test_preprocessed, join='inner', axis=1)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
# 交差検証
# clf = LogisticRegression(solver="lbfgs")
clf = LogisticRegression(solver="liblinear")
# Stratified K-Fold CV で性能を評価する
# skf = StratifiedKFold(shuffle=True, n_splits=3)
skf = StratifiedKFold(shuffle=False, n_splits=3)
scoring = {
    'acc': 'accuracy',
    # 'auc': 'roc_auc',
}
scores = cross_validate(clf, X_train_preprocessed, y_train, cv=skf, scoring=scoring)

print('Accuracy (mean):', scores['test_acc'].mean())
# print('AUC (mean):', scores['test_auc'].mean())
# 訓練
estimator = LogisticRegression()
estimator.fit(X_train_preprocessed, y_train)
X_train_preprocessed
# 予測
y_test = estimator.predict(X_test_preprocessed)
estimator.n_iter_[0]
# 書き出し
submission = pd.DataFrame({"PassengerId":test.PassengerId, "Survived":y_test})
submission.to_csv("submission.csv", index = False)
# Score: 0.78468

