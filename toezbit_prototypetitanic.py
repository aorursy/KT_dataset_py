# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# test = pd.read_csv("/kaggle/input/titanic/test.csv") #ตัวที่ต้องหาว่าใคร รอดหรือใครตาย

# test.head(3)
train = pd.read_csv("/kaggle/input/titanic/train.csv")#ชุดข้อมูลที่เอามาวิเคราะห์

train.set_index("PassengerId", inplace = True)

train["Age"].fillna(18, inplace = True)

train.head(3)
train["Survived"].value_counts() #จำนวนผู้รอดชีวิต และ เสียชีวิต
male = train["Sex"] == "male" #หาผู้ชายทั้งหมด

male_all = train[male]

len(male_all)
female = train["Sex"] == "female" #หาผู้หญิงทั้งหมด

female_all = train[female]

len(female_all)
S_S = train[["Sex","Survived"]]

sex_variableM1 = S_S["Sex"] == "male" #หาผู้ชายที่รอดชีวิต

sex_variableM2 = S_S["Survived"] == 1

male_sur = S_S[sex_variableM1&sex_variableM2]

len(male_sur)

male_sur_percent =  len(male_sur["Survived"]) / len(male_all) * 100  #เปอร์เซ็นที่ผู้ชายรอดชีวิต

male_sur_percent
sex_variableF1 = S_S["Sex"] == "female" #หาผู้หญิงที่รอดชีวิต

sex_variableF2 = S_S["Survived"] == 1

female_sur = S_S[sex_variableF1&sex_variableF2]

len(female_sur)
female_sur_percent =  len(female_sur["Survived"]) / len(female_all) *100  #เปอร์เซ็นที่ผู้หญิงรอดชีวิต

female_sur_percent
train["Fare rank"] = train["Fare"].rank(ascending = False).astype("int") #ดูอันดับค่าโดยสาร

train.sort_values(by = "Fare",ascending = False)

p1 = train["Pclass"] == 1 #หาผู้โดยสารคลาส 1ทั้งหมด

p1_all = train[p1]

len(p1_all )
P_S1 = train[["Pclass","Survived"]]

Pclass1_variableP1 = P_S1["Pclass"] == 1 #คลาสของผู้โดยสารต่อการรอดชีวิต คลาส 1

Pclass1_variableP2 = P_S1["Survived"] == 1

Pclass1_sur = P_S1[Pclass1_variableP1&Pclass1_variableP2]

len(Pclass1_sur)

class1_sur_percent =  len(Pclass1_sur["Survived"]) / len(p1_all) * 100  #เปอร์เซ็นคลาส1ที่รอด

class1_sur_percent
p2 = train["Pclass"] == 2 #หาผู้โดยสารคลาส 2ทั้งหมด

p2_all = train[p2]

len(p2_all)
P_S2 = train[["Pclass","Survived"]]

Pclass2_variableP1 = P_S2["Pclass"] == 2 #คลาสของผู้โดยสารต่อการรอดชีวิต คลาส 2

Pclass2_variableP2 = P_S2["Survived"] == 1

Pclass2_sur = P_S2[Pclass2_variableP1&Pclass2_variableP2]

len(Pclass2_sur)
class2_sur_percent =  len(Pclass2_sur["Survived"]) / len(p2_all) * 100  #เปอร์เซ็นคลาส2ที่รอด

class2_sur_percent
p3 = train["Pclass"] == 3 #หาผู้โดยสารคลาส 3ทั้งหมด

p3_all = train[p3]

len(p3_all)
P_S3 = train[["Pclass","Survived"]]

Pclass3_variableP1 = P_S3["Pclass"] == 3 #คลาสของผู้โดยสารต่อการรอดชีวิต คลาส 3

Pclass3_variableP2 = P_S3["Survived"] == 1

Pclass3_sur = P_S3[Pclass3_variableP1&Pclass3_variableP2]

len(Pclass3_sur)
class3_sur_percent =  len(Pclass3_sur["Survived"]) / len(p3_all) * 100  #เปอร์เซ็นคลาส3ที่รอด

class3_sur_percent
train["Age rank"] = train["Age"].rank(ascending = False) #ดูอันดับอายุ

train.sort_values(by = "Age",ascending = False)
train["Age"].mean() #ค่าเฉลี่ยของอายุ
age30 = train["Age"] < 30 #หาผู้โดยสารช่วงอายุต่ำกว่า 30

age30_all = train[age30]

len(age30_all)
A_S30 = train[["Age","Survived"]]

Age30_variable1 = A_S30["Age"]  < 30 # อายุต่ำกว่า 30 ที่รอด

Age30_variable2 = A_S30["Survived"] == 1

Age30_sur = A_S30[Age30_variable1 & Age30_variable2]

len(Age30_sur)
age30_sur_percent =  len(Age30_sur["Survived"]) / len(age30_all) * 100  #เปอร์เซ็นอายุต่ำกว่า 30 ที่รอด

age30_sur_percent
A_S30_50 = train[["Age","Survived"]]

age30up = train["Age"] >= 30  #หาผู้โดยสารช่วงอายุมากกว่า 30 แต่ไม่เกิน 50 ทั้งหมด

age50down = train["Age"] <= 50

age30_50_all = A_S30_50[age30up & age50down ]

len(age30_50_all)
A_S30_50 = train[["Age","Survived"]]

age30up = train["Age"] >= 30  #หาผู้โดยสารช่วงอายุมากกว่า 30 แต่ไม่เกิน 50 ที่รอด

age50down = train["Age"] <= 50

sur = A_S30_50["Survived"] == 1

age30_50_sur = A_S30_50[age30up & age50down & sur]

len(age30_50_sur)
age30_50_sur_percent =  len(age30_50_sur["Survived"]) / len(age30_50_all) * 100  #เปอร์เซ็นอายุ 30 แต่ไม่เกิน 50 ที่รอด

age30_50_sur_percent
age50 = train["Age"] > 50   #หาผู้โดยสารช่วงอายุ 50 ขึ้นไป 

age50_all = train[age50]

len(age50_all)
A_S50 = train[["Age","Survived"]]

Age50_variable1 = A_S50["Age"]  > 50 # อายุมากกว่า 50 ที่รอด

Age50_variable2 = A_S50["Survived"] == 1

Age50_sur = A_S50[Age50_variable1 & Age50_variable2]

len(Age50_sur)
age50_sur_percent =  len(Age50_sur["Survived"]) / len(age50_all) * 100  #เปอร์เซ็นอายุ 50 ขึ้นไปที่รอด

age50_sur_percent
train["SibSp"].value_counts()
Sip_3 = train["SibSp"] < 3  #หาผู้โดยสารSip ต่ำกว่า 3 

Sip3_all = train[Sip_3]

len(Sip3_all)
Sip_S3 = train[["SibSp","Survived"]]

Sip_S3_variable1 = Sip_S3["SibSp"]  < 3 # sip ต่ำกว่า 3 ที่รอด

Sip_3_variable2 = Sip_S3["Survived"] == 1

Sip3_sur = Sip_S3[Sip_S3_variable1 & Sip_3_variable2]

len(Sip3_sur)
Sip3_sur_percent =  len(Sip3_sur["Survived"]) / len(Sip3_all) * 100  #เปอร์เซ็น sip 3 ที่รอด

Sip3_sur_percent
Sip_5_1 = train["SibSp"] >= 3   #หาผู้โดยสารSip มากกว่า 3  แต่ไม่เกิน 5

Sip_5_2 = train["SibSp"] <= 5

Sip_5 = Sip_5_1&Sip_5_2

Sip3_5_all = train[Sip_5]

len(Sip3_5_all)
Sip_5S = train[["SibSp","Survived"]]

Sip3up = train["SibSp"] >= 3  #หาผู้โดยสาร sip มากกว่า 3 แต่ไม่เกิน 5 ที่รอด

Sip5down = train["SibSp"] <= 5

sur = Sip_5S["Survived"] == 1

Sip3_5_sur = Sip_5S[Sip3up & Sip5down & sur]

len(Sip3_5_sur)
Sip3_5_sur_percent =  len(Sip3_5_sur["Survived"]) / len(Sip3_5_all) * 100  #เปอร์เซ็นผู้โดยสาร sip มากกว่า 3 แต่ไม่เกิน 5 ที่รอด

Sip3_5_sur_percent
Sip_5 = train["SibSp"] > 5  #หาผู้โดยสารSip มากกว่า 5 

Sip5_all = train[Sip_5]

len(Sip5_all)
Sip_S5 = train[["SibSp","Survived"]]

Sip_S5_variable1 = Sip_S5["SibSp"]  > 5 # sip มากกว่า 5 ที่รอด

Sip_5_variable2 = Sip_S5["Survived"] == 1

Sip5_sur = Sip_S5[Sip_S5_variable1 & Sip_5_variable2]

len(Sip5_sur)
Sip5_sur_percent =  len(Sip5_sur["Survived"]) / len(Sip5_all) * 100  #เปอร์เซ็นผู้โดยสาร sip มากกว่า 5 ที่รอด

Sip5_sur_percent
train["Parch"].value_counts()
Parch_3 = train["Parch"] < 3  #หาผู้โดยสารParch ต่ำกว่า 3 

Parch3_all = train[Parch_3]

len(Parch3_all)
Parch_S3 = train[["Parch","Survived"]]

Parch_S3_variable1 = Parch_S3["Parch"]  < 3 # Parch ต่ำกว่า 3 ที่รอด

Parch_S3_variable2 = Parch_S3["Survived"] == 1

Parch3_sur = Parch_S3[Parch_S3_variable1 & Parch_S3_variable2]

len(Parch3_sur)
Parch3_sur_percent =  len(Parch3_sur["Survived"]) / len(Parch3_all) * 100  #เปอร์เซ็นผู้โดยสาร Parch ต่ำกว่า 3 ที่รอด

Parch3_sur_percent
Parch_5_1 = train["Parch"] >= 3   #หาผู้โดยสารParch ต่ำ 3  แต่ไม่เกิน 5

Parch_5_2 = train["Parch"] <= 5

Parch3_5 = Parch_5_1&Parch_5_2

Parch3_5_all = train[Parch3_5]

len(Parch3_5_all)
Parch_5S = train[["Parch","Survived"]]

Parch3up = train["Parch"] >= 3  #หาผู้โดยสาร Parch มากกว่า 3 แต่ไม่เกิน 5 ที่รอด

Parch5down = train["Parch"] <= 5

sur = Parch_5S["Survived"] == 1

Parch3_5_sur = Parch_5S[Parch3up & Parch5down & sur]

len(Parch3_5_sur)
Parch3_5_sur_percent =  len(Parch3_5_sur["Survived"]) / len(Parch3_5_all) * 100  #เปอร์เซ็นผู้โดยสาร Parch มากว่า 3 ไม่เกิน 5

Parch3_5_sur_percent
Parch_5 = train["Parch"] > 5  #หาผู้โดยสาร Parch มากกว่า 5

Parch_5_all = train[Parch_5]

len(Parch_5_all)
Parch_S5 = train[["Parch","Survived"]]

Parch_S5_variable1 = Parch_S5["Parch"]  > 5 # Parch มากกว่า 5 ที่รอด

Parch_5_variable2 = Parch_S5["Survived"] == 1

Parch5_sur = Parch_S5[Parch_S5_variable1 & Parch_5_variable2]

len(Parch5_sur)
Parch5_sur_percent =  len(Parch5_sur["Survived"]) / len(Parch_5_all) * 100  #เปอร์เซ็นผู้โดยสาร Parch มากว่า   5 ที่รอด

Parch5_sur_percent
test = pd.read_csv("/kaggle/input/titanic/test.csv", index_col = "PassengerId") #ตัวที่ต้องหาว่าใคร รอดหรือใครตาย

test["Age"].fillna(18, inplace = True)

test.head(3)
def Who_Survive(row):

    pclass = row[0]

    sex = row[2]

    age = row[3]

    sibsp = row[4]

    parch = row[5]

    

    if   sibsp >=3 and sibsp <=5 and parch <3 and pclass == "2":

        return "Survive"

    elif sex == "female" and parch <3 and sibsp <3 and pclass == "1":

        return "Survive"

    elif pclass == "2" and age <30 and sibsp <3 and parch <3:

        return "Survive"

    elif parch <3 and age >=30 and age <=50 :

        return "Survive"

    elif sibsp >=3 and sibsp <=5 and parch >=3 and parch <=5 and pclass == "1" :

        return "Survive"

    else:

        return "RIP"
test["Result"] = test.apply(Who_Survive, axis = "columns")
test
test["Result"].value_counts()


    