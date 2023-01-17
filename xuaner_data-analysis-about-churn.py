import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()
print(data.info())
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors='coerce')
data.info()
data.dropna(inplace = True)
data.info()
plt.figure(figsize = (11,5))
data.MonthlyCharges.plot.hist(color ='lightgreen')
#plt.hist(data.TotalCharges,10)
plt.title('MonthlyCharges', size = 15, color = 'purple')
plt.xlabel('Charges_groups', size = 13, color = 'b')
plt.xticks(size = 13, color = 'g')
plt.yticks(size = 13,color = 'g')
plt.show()
charges_group = [[0,20]]
for i in range(6):
    charges_group.append([i+20 for i in charges_group[len(charges_group) - 1]])
per = []
per_no = []
for i in range(6):
    i_ = data.loc[(data.MonthlyCharges >= i*20) & (data.MonthlyCharges < (i+1)*20)]
    Churn_per = len(i_[i_.Churn == 'Yes'])*100/len(i_)
    Not_Churn_per = len(i_[i_.Churn =='No'])*100/len(i_)
    per.append(Churn_per)
    per_no.append(Not_Churn_per)
df = pd.DataFrame({'Churn_per': per,'Not_Churn_per': per_no})
plt.figure(figsize = (15,8))
p1 = plt.bar(range(6), tuple(per), 0.56, color='#45cea2' )
p2 = plt.bar(range(6), tuple(per_no), 0.56, bottom = tuple(per), color = '#fdd470')
plt.xticks(range(6), charges_group, size = 13, color = 'g')
plt.yticks(size = 13,color = 'g')
plt.xlabel('Monthlycharges_group', size =13, color = 'blue')
plt.ylabel('per', size = 13, color = 'blue')
plt.title('Churn per and Notchurn per of customer who has no partner',size = 13, color = 'c')
plt.show()
plt.figure(figsize = (11,5))
data.TotalCharges.plot.hist(color ='lightgreen')
#plt.hist(data.TotalCharges,10)
plt.title('TotalCharges', size = 15, color = 'purple')
plt.xticks(size = 13, color = 'g')
plt.yticks(size = 13,color = 'g')
plt.show()
charges_group = [[0,1000]]
for i in range(9):
    charges_group.append([i+1000 for i in charges_group[len(charges_group) - 1]])
per = []
per_no = []
for i in range(9):
    i_ = data.loc[(data.TotalCharges >= i*1000) & (data.TotalCharges < (i+1)*1000) & (data.Partner == 'Yes')]
    Churn_per = len(i_[i_.Churn == 'Yes'])*100/len(i_)
    Not_Churn_per = len(i_[i_.Churn =='No'])*100/len(i_)
    per.append(Churn_per)
    per_no.append(Not_Churn_per)
df = pd.DataFrame({'Churn_per': per,'Not_Churn_per': per_no})
plt.figure(figsize = (15,8))
p1 = plt.bar(range(9), tuple(per), 0.56, color='#45cea2' )
p2 = plt.bar(range(9), tuple(per_no), 0.56, bottom = tuple(per), color = '#fdd470')
plt.xticks(range(9), charges_group, size = 13, color = 'g')
plt.yticks(size = 13,color = 'g')
plt.xlabel('Total_charges_group', size =13, color = 'blue')
plt.ylabel('per', size = 13, color = 'blue')
plt.title('Churn per and NotChurn per of customer who has partner',size = 16, color = 'c')
plt.show()

charges_group = [[0,1000]]
for i in range(9):
    charges_group.append([i+1000 for i in charges_group[len(charges_group) - 1]])
per = []
per_no = []
for i in range(9):
    i_ = data.loc[(data.TotalCharges >= i*1000) & (data.TotalCharges < (i+1)*1000) & (data.Partner == 'No')]
    Churn_per = len(i_[i_.Churn == 'Yes'])*100/len(i_)
    Not_Churn_per = len(i_[i_.Churn =='No'])*100/len(i_)
    per.append(Churn_per)
    per_no.append(Not_Churn_per)
df = pd.DataFrame({'Churn_per': per,'Not_Churn_per': per_no})
plt.figure(figsize = (15,8))
p1 = plt.bar(range(9), tuple(per), 0.56, color='#45cea2' )
p2 = plt.bar(range(9), tuple(per_no), 0.56, bottom = tuple(per), color = '#fdd470')
plt.xticks(range(9), charges_group, size = 13, color = 'g')
plt.yticks(size = 13,color = 'g')
plt.xlabel('Total_charges_group', size =13, color = 'blue')
plt.ylabel('per', size = 13, color = 'blue')
plt.title('Churn per and Notchurn per of customer who has no partner',size = 16, color = 'c')
plt.show()
charges_group = [[0,1000]]
for i in range(9):
    charges_group.append([i+1000 for i in charges_group[len(charges_group) - 1]])
per = []
per_no = []
for i in range(9):
    i_ = data.loc[(data.TotalCharges >= i*1000) & (data.TotalCharges < (i+1)*1000)]
    Churn_per = len(i_[i_.Churn == 'Yes'])*100/len(i_)
    Not_Churn_per = len(i_[i_.Churn =='No'])*100/len(i_)
    per.append(Churn_per)
    per_no.append(Not_Churn_per)
df = pd.DataFrame({'Churn_per': per,'Not_Churn_per': per_no})
plt.figure(figsize = (15,8))
p1 = plt.bar(range(9), tuple(per), 0.56, color='#45cea2' )
p2 = plt.bar(range(9), tuple(per_no), 0.56, bottom = tuple(per), color = '#fdd470')
plt.xticks(range(9), charges_group, size = 13, color = 'g')
plt.yticks(size = 13,color = 'g')
plt.xlabel('Total_charges_group', size =13, color = 'blue')
plt.ylabel('per', size = 13, color = 'blue')
plt.title('Churn per and Notchurn per of customer',size = 15, color = 'c')
plt.show()
plt.figure(figsize = (8,8))
sns.set(style = 'whitegrid')
sns.countplot(data.SeniorCitizen, hue = data.Churn )
plt.show()
plt.figure(figsize = (10,8))

sns.set(style = 'whitegrid')
sns.boxplot(x = data.SeniorCitizen, y = data.MonthlyCharges, hue = data.Churn)

plt.title('Total Revenue by Seniors and Non-Seniors', color = 'orange', size = 15)
plt.figure(figsize = (10,5))
sns.set(style = 'whitegrid')
sns.countplot(data.PaymentMethod, hue = data.Churn )
plt.figure(figsize = (15,5))
sns.countplot(data.InternetService, hue = data.Churn)
numerics = data[['tenure','MonthlyCharges', 'TotalCharges', 'Churn']]
plt.figure(figsize = (10,10))
sns.regplot(x = 'tenure', y = 'TotalCharges', data = numerics,color = 'c')
plt.title('Relationship between loyalty months and total revenue',color = 'orange',size = 15)