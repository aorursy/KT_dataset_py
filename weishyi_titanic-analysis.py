%matplotlib inline

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



#feat_coloums = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

train.head()
pcls_sur = train.loc[:,["Survived","Pclass"]]

#pcls_sur.fillna(method="ffill",inplace=True)

pcls_sur.dropna(inplace=True)

pcls_sur.head()



non_sur_lst = []

survive_lst = []

for pcl in set(pcls_sur.Pclass.values):

    p1_non_sur = len([i for i in pcls_sur.values if i[0] == 0 and i[1] == pcl] )

    p1_sur = len([i for i in pcls_sur.values if i[0] == 1 and i[1] == pcl] )

    non_sur_lst.append(p1_non_sur)

    survive_lst.append(p1_sur)



width = 0.5

ind = np.arange(len(non_sur_lst))



plt.subplots_adjust(wspace=0.5)

plt.subplot(1,2,1)

p1 = plt.bar(ind,survive_lst,width,color='r')

p2 = plt.bar(ind,non_sur_lst,width,color='y',bottom=survive_lst)

plt.ylabel("People")

plt.title("Pclass & Survival")

plt.xticks(ind,("P1","P2","P3"))

plt.yticks(np.arange(0,600,50))

plt.legend((p2[0],p1[0]),("non-sur","survive"))



#每个级别的生还率

survives = np.array(survive_lst)

non_surs = np.array(non_sur_lst)

sur_rat = survives / (survives + non_surs)

plt.subplot(1,2,2)

plt.bar(ind,sur_rat)

plt.title("Survival Ratio")

plt.ylabel("ratio")

plt.yticks(np.arange(0,1,0.2))

plt.xticks(ind,("P1","P2","P3"))

plt.show()
#Sex & Survival

raw_sur = train.loc[:,["Survived","Sex"]]

#raw_sur.fillna(method="ffill",inplace=True)

raw_sur.dropna(inplace=True)



non_sur_lst = []

survive_lst = []

for pcl in set(raw_sur.Sex.values):

    p1_non_sur = len([i for i in raw_sur.values if i[0] == 0 and i[1] == pcl] )

    p1_sur = len([i for i in raw_sur.values if i[0] == 1 and i[1] == pcl] )

    non_sur_lst.append(p1_non_sur)

    survive_lst.append(p1_sur)



width = 0.5

ind = np.arange(len(non_sur_lst))



plt.subplots_adjust(wspace=0.5)

plt.subplot(1,2,1)

p1 = plt.bar(ind,survive_lst,width,color='r')

p2 = plt.bar(ind,non_sur_lst,width,color='y',bottom=survive_lst)

plt.ylabel("People")

plt.title("Sex & Survival")

plt.xticks(ind,("Male","Female"))

plt.yticks(np.arange(0,800,100))

plt.legend((p2[0],p1[0]),("non-sur","survive"))



#每个级别的生还率

survives = np.array(survive_lst)

non_surs = np.array(non_sur_lst)

sur_rat = survives / (survives + non_surs)

plt.subplot(1,2,2)

plt.bar(ind,sur_rat)

plt.title("Survival Ratio")

plt.ylabel("ratio")

plt.yticks(np.arange(0,1,0.2))

plt.xticks(ind,("Male","Female"))

plt.show()
def ageSeg(age):

    return int(age / 20)



#Age distribute

raw_ages = train.loc[:,["Survived","Age"]]

#raw_ages.fillna(method="ffill",inplace=True)

raw_ages.dropna(inplace=True)

ages = raw_ages.Age.values

plt.subplots_adjust(hspace=1.,wspace=1.)

plt.subplot(221)

plt.title("Age Distribute")

plt.xlabel("age")

plt.ylabel("people")

plt.hist(ages)



#Age & Survival

age_grp = ["(~-20)","(20-40)","(40-60)","(60-~)"]

age_dict = {}

for n in range(len(age_grp)):

     age_dict[age_grp[n]] = [i for i in raw_ages.values if ageSeg(i[1]) == n]



plt.subplot(222)

non_sur_lst = []

survival_lst = []

for k in age_dict:

    non_sur_lst.append(len([i for i in age_dict[k] if i[0] == 0]))

    survival_lst.append(len([i for i in age_dict[k] if i[0] == 1]))

ind = np.arange(len(non_sur_lst))

width = 0.35



p1 = plt.bar(ind,survival_lst,width=width,color='r')

p2 = plt.bar(ind,non_sur_lst,width=width,color='y',bottom=survival_lst)

plt.title("Age & Survival")

plt.xticks(ind,age_grp)

plt.legend((p2[0],p1[0]),("non-sur","survival"))



plt.subplot(223)

age_grp_count = np.array(non_sur_lst) + np.array(survival_lst)

plt.pie(age_grp_count,labels=age_grp,labeldistance = 1.1,startangle = 90,pctdistance = 2.,autopct='%1.2f%%')



plt.show()
#SibSp & Survival

raw_sur = train.loc[:,["Survived","SibSp"]]

#raw_sur.fillna(method="ffill",inplace=True)

raw_sur.dropna(inplace=True)



non_sur_lst = []

survive_lst = []

for pcl in set(raw_sur.SibSp.values):

    p1_non_sur = len([i for i in raw_sur.values if i[0] == 0 and i[1] == pcl] )

    p1_sur = len([i for i in raw_sur.values if i[0] == 1 and i[1] == pcl] )

    non_sur_lst.append(p1_non_sur)

    survive_lst.append(p1_sur)



width = 0.5

ind = np.arange(len(non_sur_lst))

plt.subplots_adjust(wspace=0.5)

plt.subplot(1,2,1)

p1 = plt.bar(ind,survive_lst,width,color='r')

p2 = plt.bar(ind,non_sur_lst,width,color='y',bottom=survive_lst)

plt.ylabel("People")

plt.title("SibSp & Survived")

plt.xticks(ind,[str(i) for i in range(len(ind))])

plt.yticks(np.arange(0,800,100))

plt.legend((p2[0],p1[0]),("non-sur","survive"))



#每个级别的生还率

survives = np.array(survive_lst)

non_surs = np.array(non_sur_lst)

sur_rat = survives / (survives + non_surs)

plt.subplot(1,2,2)

plt.bar(ind,sur_rat)

plt.title("Survival Ratio")

plt.ylabel("ratio")

plt.yticks(np.arange(0,1,0.2))

plt.xticks(ind,[str(i) for i in range(len(ind))])

plt.show()
#Parch & Survival

raw_sur = train.loc[:,["Survived","Parch"]]

#raw_sur.fillna(method="ffill",inplace=True)

raw_sur.dropna(inplace=True)



non_sur_lst = []

survive_lst = []

for pcl in set(raw_sur.Parch.values):

    p1_non_sur = len([i for i in raw_sur.values if i[0] == 0 and i[1] == pcl] )

    p1_sur = len([i for i in raw_sur.values if i[0] == 1 and i[1] == pcl] )

    non_sur_lst.append(p1_non_sur)

    survive_lst.append(p1_sur)



width = 0.5

ind = np.arange(len(non_sur_lst))

plt.subplots_adjust(wspace=0.5)

plt.subplot(1,2,1)

p1 = plt.bar(ind,survive_lst,width,color='r')

p2 = plt.bar(ind,non_sur_lst,width,color='y',bottom=survive_lst)

plt.ylabel("People")

plt.title("Parch & Survived")

plt.xticks(ind,[str(i) for i in range(len(ind))])

plt.yticks(np.arange(0,800,100))

plt.legend((p2[0],p1[0]),("non-sur","survive"))



#每个级别的生还率

survives = np.array(survive_lst)

non_surs = np.array(non_sur_lst)

sur_rat = survives / (survives + non_surs)

plt.subplot(1,2,2)

plt.bar(ind,sur_rat)

plt.title("Survival Ratio")

plt.ylabel("ratio")

plt.yticks(np.arange(0,1,0.2))

plt.xticks(ind,[str(i) for i in range(len(ind))])

plt.show()
#Fare distribute

raw_dat = train.loc[:,["Survived","Fare"]]

#raw_sur.fillna(method="ffill",inplace=True)

raw_dat.dropna(inplace=True)



fare = np.array(raw_dat.iloc[:,1].values)

plt.subplots_adjust(wspace = 0.5,hspace = 0.5)

plt.subplot(221)

plt.title("Fare distribute")

plt.hist(fare)

#print("min = ",fare.min()," max=",fare.max())



fare_grp = ["~-100","100-200","200-300","300-~"]

fare_dict = {}

for k in range(len(fare_grp)):

    fare_dict[fare_grp[k]] = [i for i in raw_dat.values if int(i[1] / 100) == k]

non_sur = []

be_save = []

width = 0.35

for k in fare_dict:

    non_sur.append(len([i for i in fare_dict[k] if i[0] == 0]))

    be_save.append(len([i for i in fare_dict[k] if i[0] == 1]))

ind = np.arange(len(non_sur))

plt.subplot(222)

p1 = plt.bar(ind,be_save,color='r',width=width)

p2 = plt.bar(ind,non_sur,color='y',width=width,bottom=be_save)

plt.ylabel("People")

plt.title("Fare & Survived")

plt.xticks(ind,fare_grp,rotation=17)

plt.legend((p2[0],p1[0]),("non-sur","survive"))

plt.show()
#Embarked & Survived

raw_dat = train.loc[:,["Survived","Embarked"]]

#raw_sur.fillna(method="ffill",inplace=True)

raw_dat.dropna(inplace=True)



#Embarked distribute

embarked_count = []

embarked_dict = {}

embarked_set = set(raw_dat.Embarked.values)

for k in embarked_set:

    lst = [i for i in raw_dat.values if i[1] == k]

    embarked_count.append(len(lst))

    embarked_dict[k] = lst

plt.subplot(221)

plt.pie(embarked_count,labels=list(embarked_set),autopct='%1.2f%%')

plt.title("Embarked Distribute")



non_sur = []

be_save = []

for k in embarked_dict:

    non_sur.append(len([i for i in embarked_dict[k] if i[0] == 0]))

    be_save.append(len([i for i in embarked_dict[k] if i[0] == 1]))

ind = np.arange(len(non_sur))

plt.subplot(222)

p1 = plt.bar(ind,be_save,color='r',width=width)

p2 = plt.bar(ind,non_sur,color='y',width=width,bottom=be_save)

plt.ylabel("People")

plt.title("Embarked & Survived")

plt.xticks(ind,list(embarked_set),rotation=17)

plt.legend((p2[0],p1[0]),("non-sur","survive"))

plt.show()