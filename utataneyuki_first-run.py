import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

%matplotlib inline

font = {"family":"HGMaruGothicMPRO"}

matplotlib.rc("font",**font)

import seaborn as sns

from pandas import Series,DataFrame

import numpy as np
import os

print(os.listdir("../input"))
#データの取り込み（念のため国情報も）

train = pd.read_csv("../input/exam-for-students20200129/train.csv")

test = pd.read_csv("../input/exam-for-students20200129/test.csv")

country = pd.read_csv("../input/exam-for-students20200129/country_info.csv")
train.info()
#目的変数確認

train["ConvertedSalary"].head()
plt.hist(train["ConvertedSalary"])
#対数処理

train["ConvertedSalary"] = np.log1p(train[["ConvertedSalary"]])
#処理後確認用

train["ConvertedSalary"].head()
plt.hist(train["ConvertedSalary"])
#二つのファイルに同じ特徴量エンジニアリングを行う為、一旦結合

##見分ける為の処理記述

train["hantei"] = "train"

test["hantei"] = "test"

full_data= pd.concat([train, test])
#結合確認用

full_data
#国データの結合

full_data = full_data.merge(country, on='Country', how='left').set_index(full_data.index)
full_data
#各列のユニーク数を見る

print(train.nunique())
#没

#for i in range(0,10):

#    print(train[i].nunique())
#試運転

retsumei = []

retsumei = train.columns
#試運転

for i in retsumei:

    print(i,train[i].nunique())
#fullのほうで稼働

full_retsumei = []

full_retsumei = full_data.columns

for j in full_retsumei:

    print(j,full_data[j].nunique(),full_data[j].dtype)
#出現頻度が多くて、objectで入っているものを探さす

full_retsumei = []

full_retsumei = full_data.columns

for j in full_retsumei:

    if full_data[j].nunique() >50:

        if full_data[j].dtype =="object":

            print(j,full_data[j].nunique(),full_data[j].dtype)
#％で入っているデータが、csv形式であるために.が使えず

#,を代用している可能性がある為、置換と型変換を行う。
#目視で見た限りこの列群

"""

Pop. Density (per sq. mi.)

Coastline (coast/area ratio)

Net migration

Infant mortality (per 1000 births)

Literacy (%)

Phones (per 1000)

Arable (%)

Crops (%)

Other (%)

Birthrate

Deathrate

Agriculture

Industry

Service

"""
#試運転

print(full_data['Service'].str.replace(',', '.'))
#,を.に　typeをfloatに

#まず初めに置換

henkan = ["Pop. Density (per sq. mi.)","Coastline (coast/area ratio)","Net migration","Infant mortality (per 1000 births)"

          ,"Literacy (%)","Phones (per 1000)","Arable (%)","Crops (%)","Other (%)"

          ,"Birthrate","Deathrate","Agriculture","Industry","Service"]

for k in henkan:

    #print(full_data[k].str.replace(',', '.'))

    print(k)

    full_data[k] = full_data[k].str.replace(',', '.')
#確認

#full_data
#型変換

#まだobjectであることを確認

#試運転

print(full_data['Service'].dtype)
#試運転

print(full_data['Service'].astype("float64"))
for l in henkan:

    print(full_data[l].astype("float64"))
#エラー検出した箇所

for l in henkan:

    full_data[l] = full_data[l].astype("float64")

#print(full_data['Service'].dtype)
#print(full_data["Pop. Density (per sq. mi.)"])
#print(full_data["Pop. Density (per sq. mi.)"].astype("float64"))
for j in full_retsumei:

    if full_data[j].nunique() >50:

        if full_data[j].dtype =="object":

            print(j,full_data[j].nunique(),full_data[j].dtype)
#Countryはテキスト

#CurrencySymbolも三文字のテキスト（米ドルなどを示した三文字？）

#DevTypeも文字

#FrameworkWorkedWithも文字

#RaceEthnicityも文字（人種）

#どれもcsvよけではない為先ほどの処理は終了
##Ageが範囲指定になっているので、他にもなっていないか確認

for j in full_retsumei:

    if full_data[j].nunique() <30 and full_data[j].nunique() > 2:

        if full_data[j].dtype =="object":

            print(j,full_data[j].nunique(),full_data[j].dtype)
"""

AdsActions 15 object  普通のテキスト

Age 7 object　数値を文字で表現

CareerSatisfaction 7 object

CheckInCode 6 object

CompanySize 8 object

Currency 19 object

EducationParents 9 object

Employment 6 object

ErgonomicDevices 15 object

FormalEducation 9 object

Gender 15 object

HopeFiveYears 7 object

JobSatisfaction 7 object

SexualOrientation 14 object

StackOverflowJobsRecommend 11 object

StackOverflowParticipate 6 object

StackOverflowRecommend 11 object

StackOverflowVisit 6 object

TimeAfterBootcamp 8 object

TimeFullyProductive 6 object

UndergradMajor 12 object

UpdateCV 8 object

WakeTime 11 object

YearsCoding 11 object

YearsCodingProf 11 object

Region 11 object

Climate 6 object

"""

#年齢など数値的強弱がありそうなテキストを数値変換

"""

Age 7 object



CompanySize

HopeFiveYears

YearsCoding

YearsCodingProf



AdsAgreeDisagree1 5 object

AdsAgreeDisagree2 5 object

AdsAgreeDisagree3 5 object

AgreeDisagree1 5 object

AgreeDisagree2 5 object

AgreeDisagree3 5 object



YearsCoding 11 object

YearsCodingProf 11 object



HopeFiveYears 7 object

"""
#Ageの要素を見る

full_data["Age"].value_counts() 

#age = ["25 - 34 years old"

#,"18 - 24 years old"

#,"35 - 44 years old"

#,"45 - 54 years old"

#,"55 - 64 years old"

#,"Under 18 years old"

#,"65 years or older"]

#年齢置換

full_data["Age"] = full_data["Age"].str.replace('Under 18 years old', '1')

full_data["Age"] = full_data["Age"].str.replace('18 - 24 years old', '2')

full_data["Age"] = full_data["Age"].str.replace('25 - 34 years old', '3')

full_data["Age"] = full_data["Age"].str.replace('35 - 44 years old', '4')

full_data["Age"] = full_data["Age"].str.replace('45 - 54 years old', '5')

full_data["Age"] = full_data["Age"].str.replace('55 - 64 years old', '6')

full_data["Age"] = full_data["Age"].str.replace('65 years or older', '7')
full_data['Age'].astype("float64")
full_data["Age"].value_counts() 
full_retsumei = []

full_retsumei = full_data.columns

ordinal = []

for j in full_retsumei:

    if full_data[j].dtype =="object":

        print(j,full_data[j].nunique(),full_data[j].dtype)

        ordinal.append(j)
ordinal
import category_encoders as ce

oe = ce.OrdinalEncoder(cols=ordinal, return_df=False)

full_data[ordinal] = oe.fit_transform(full_data[ordinal])
#欠損値を正の数以外で埋めておく

full_data.fillna(-1, inplace=True)
full_data
#1行も入らない

#→オーディナルでつぶしてしまった

#X_train = full_data[full_data['hantei'] == "train"]

#X_test = full_data[full_data['hantei'] == "test"]



#X_train.drop(['hantei'], axis=1, inplace=True)

#X_test.drop(['hantei'], axis=1, inplace=True)



#1がトレイン　2がテスト

full_data["hantei"].value_counts()
len(full_data[full_data['hantei'] == 1])
X_train = full_data[full_data['hantei'] == 1]

X_test = full_data[full_data['hantei'] == 2]



X_train.drop(['hantei'], axis=1, inplace=True)

X_test.drop(['hantei'], axis=1, inplace=True)
X_train1
import lightgbm as lgb

from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split

y_train = X_train["ConvertedSalary"]

X_train2 = X_train.drop(["ConvertedSalary"], axis=1)

X_test = X_test.drop(["ConvertedSalary"], axis=1)

#X_test[["ConvertedSalary"]] = ""

X_train2.sort_index(axis=1, ascending=False)

X_test.sort_index(axis=1, ascending=False)
X_train1,X_test1,y_train,y_test = train_test_split(X_train2,y_train)
model = lgb.LGBMRegressor()

model.fit(X_train1, y_train)
pred = model.predict(X_test1)
len(pred)
pred1 = np.expm1(pred)
pred1[1]
pred = model.predict(X_test)
#print(X_test.columns)

print(X_train1.columns)
for h in X_train1.columns:

    print(h)
pred1 = np.expm1(pred)
submission = pd.read_csv('../input/exam-for-students20200129/sample_submission.csv', index_col=0)

submission['ConvertedSalary'] = pred1

submission.to_csv('submission.csv')