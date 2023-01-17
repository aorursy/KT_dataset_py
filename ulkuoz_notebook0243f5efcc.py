import os
import sys
os.getcwd()
os.chdir('/kaggle/input/cargo-data/Gemi/GemiSayısı')
import pandas as pd
import numpy as np

files=os.listdir()
files
My_df1=pd.DataFrame()
i=0
for file in files:
        df=pd.read_excel(file,usecols=list(range(13)))
        df=df.drop([0,1,2],axis=0)
        df.columns=["Liman", "Türk_Farklı_Sayı","Türk_Farklı_Ton","Yabancı_Farklı_Sayı","Yabancı_Farklı_Ton",
            "Toplam_Farklı_Sayı","Toplam_Ton_Farklı","Toplam_Türk_Sayı","Toplam_Türk_Ton","Toplam_Gemi_Yabancı_Sayı",
            "Toplam_Gemi_Yabancı_Ton","Toplam_Gemi","Toplam_Ton"]
        df["Yıl"]=file[:4]
        My_df1=My_df1.append(df, ignore_index=True)
    


My_df1.head()
os.chdir('/kaggle/input/cargo-data/Gemi')
file2=os.listdir()
file2
del file2[7]
file2
My_df2=pd.DataFrame()
i=0
for file in file2:
        df=pd.read_excel(file,usecols=list(range(22)))
        df=df.drop([0,1,2],axis=0)
        df=df.iloc[:,[0,4,5,6,11,12,13,21]]
        df.columns=["Liman", "Toplam_İhracat","Kabotaj_Yükleme","Transit_Yükleme",
                   "Toplam_İthalat","Kabotaj_Boşaltma","Transit_Boşaltma","Toplam_elleçleme"]
        df["Yıl"]=file[:4]
        My_df2=My_df2.append(df, ignore_index=True)
        i +=1
    
My_df2.shape
Merged_Data=pd.merge(My_df1,My_df2,how="right", on=["Liman","Yıl"])
Merged_Data.shape
Liman=Merged_Data["Liman"].value_counts()>1
My_Liman=Liman.index[Liman]

My_Liman=My_Liman[My_Liman!="TOPLAM"]
Merged_Data=Merged_Data.dropna()
Merged_Data.shape
Merged_Data=Merged_Data.loc[Merged_Data['Liman'].isin(My_Liman)]

Merged_Data1=Merged_Data[['Türk_Farklı_Sayı', 'Türk_Farklı_Ton',
       'Yabancı_Farklı_Sayı', 'Yabancı_Farklı_Ton', 'Toplam_Farklı_Sayı',
       'Toplam_Ton_Farklı', 'Toplam_Türk_Sayı', 'Toplam_Türk_Ton',
       'Toplam_Gemi_Yabancı_Sayı', 'Toplam_Gemi_Yabancı_Ton',
       'Toplam_Gemi', 'Toplam_Ton', 'Toplam_İhracat', 'Kabotaj_Yükleme',
       'Transit_Yükleme', 'Toplam_İthalat', 'Kabotaj_Boşaltma',
       'Transit_Boşaltma', 'Toplam_elleçleme',"Yıl"]].astype(int)

Merged_Data1["Liman"]=Merged_Data["Liman"]
Merged_Data1=Merged_Data1[Merged_Data1["Toplam_elleçleme"]>0]
Merged_Data1.describe()
Merged_Data1.info()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set()
plt.figure(figsize=(20,30))

sns.pairplot(Merged_Data1, plot_kws=dict(alpha=.1, edgecolor='none'),
            x_vars=Merged_Data1.columns.values[:5],
            y_vars=["Toplam_elleçleme"] 
            )

sns.pairplot(Merged_Data1, plot_kws=dict(alpha=.1, edgecolor='none'),
            x_vars=Merged_Data1.columns.values[5:10],
            y_vars=["Toplam_elleçleme"] 
            )
sns.pairplot(Merged_Data1, plot_kws=dict(alpha=.1, edgecolor='none'),
            x_vars=Merged_Data1.columns.values[10:15],
            y_vars=["Toplam_elleçleme"] 
            )
sns.pairplot(Merged_Data1, plot_kws=dict(alpha=.1, edgecolor='none'),
            x_vars=Merged_Data1.columns.values[15:19],
            y_vars=["Toplam_elleçleme"] 
            )
Merged_Data1[["Türk_Farklı_Sayı","Yabancı_Farklı_Sayı","Toplam_elleçleme"]].hist(bins=50, figsize=(13,7))
Data=Merged_Data1[["Türk_Farklı_Sayı","Yabancı_Farklı_Sayı","Toplam_elleçleme"]]
Data.describe()
import statsmodels.formula.api as smf

results = smf.ols('Toplam_elleçleme ~ Türk_Farklı_Sayı + Yabancı_Farklı_Sayı', data=Data).fit()
print(results.summary())
pred_val = results.fittedvalues.copy()
true_val = Data['Toplam_elleçleme'].values.copy()
residual = true_val - pred_val
plt.style.use('seaborn-white')
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(true_val, pred_val, marker="o", edgecolor="black", color="red")
ax.set_title("Predicted Values vs Real Values")
ax.set_xlabel("Real Output")
ax.set_ylabel("Predicted Output")

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(residual, pred_val, marker="o", edgecolor="black", color="red")
ax.set_title("Residual Plot")
ax.set_xlabel("Residual")
ax.set_ylabel("Predicted Output")
import statsmodels.stats.api as sms

sms.linear_harvey_collier(results)
import statsmodels

_, pval, __, f_pval = statsmodels.stats.diagnostic.het_breuschpagan(residual, Data[['Türk_Farklı_Sayı', 'Yabancı_Farklı_Sayı']])
pval, f_pval
import scipy as sp
fig, ax = plt.subplots(figsize=(6,6))
_, (__, ___, r) = sp.stats.probplot(residual, plot=ax, fit=True,)
r**2
Port_Data=Merged_Data1.groupby("Liman")["Toplam_elleçleme"].sum()
Port_Data=Port_Data.sort_values(ascending=False)
plt.subplots(figsize=(12,12))
sns.barplot(x=Port_Data.index, y=Port_Data)
plt.xticks(rotation=90)

#bar.set_xticklabels(bar.get_xticklabels(),rotation=90)
Yabancı_Farklı=Merged_Data1.groupby("Liman")["Yabancı_Farklı_Sayı"].sum()
Port_Data=Merged_Data1.groupby("Liman")["Toplam_elleçleme"].sum()

Data2=pd.DataFrame(Yabancı_Farklı)
Data2["Toplam_elleçleme"]=Port_Data
Data2=Data2.sort_values(by="Toplam_elleçleme",ascending=False)
Data2.info()
plt.style.use('seaborn-white')

colors = sns.color_palette()
plt.figure(figsize=(12,8))
plt.xticks(x=range(54),label=Data2.index,rotation='vertical')
# Setup the dual y-axes
ax1 = plt.axes()
ax2 = ax1.twinx()   ## We say use same x but different y scale...!!!!!!
# Plot the linear regression data
ax1.bar(Data2.index,Data2["Toplam_elleçleme"], width=1, color="wheat")
# Plot the regularization data sets
ax2.plot(Data2.index,Data2["Yabancı_Farklı_Sayı"],color='green', marker='o', linestyle='dashed',
          linewidth=2, markersize=8)

# Customize axes scales
ax1.set_ylim(0, 5e8)
ax2.set_ylim(0, 7500)
# Combine the legends

h1, l1 = ax1.get_legend_handles_labels()  ### h1 göstergesi l1 label adı
h2, l2 = ax2.get_legend_handles_labels()  ### h2 göstergesi l2 label adı
ax1.legend(h1+h2, l1+l2)

ax1.set(xlabel='Ports',ylabel='Total Output-Ton')
ax2.set(ylabel='Foreign Ship Call')



Yabancı_Farklı=Merged_Data1.groupby("Liman")["Yabancı_Farklı_Sayı"].sum()
Port_Data=Merged_Data1.groupby("Liman")["Toplam_elleçleme"].sum()
Turk_Farklı=Merged_Data1.groupby("Liman")["Türk_Farklı_Sayı"].sum()

Data3=pd.DataFrame(Yabancı_Farklı)
Data3["Toplam_elleçleme"]=Port_Data
Data3["Türk_Farklı_Sayı"]=Turk_Farklı
Data3=Data3.sort_values(by="Toplam_elleçleme",ascending=False)


Data3.shape

plt.style.use('seaborn-white')
Data=Data3
x=range(53*150)[::150]

colors = sns.color_palette()
plt.figure(figsize=(12,8))

# Setup the dual y-axes
ax3 = plt.axes()
ax4 = ax3.twinx()   ## We say use same x but different y scale...!!!!!!


# Plot the linear regression data
ax3.bar(x,Data["Toplam_elleçleme"], width=70, edgecolor="black", facecolor="wheat",
       align="center")
#plt.xticks(x,label=Data.index,rotation='vertical')


# Plot the regularization data sets
ax4.plot(x,Data["Yabancı_Farklı_Sayı"],color='green', marker='o', linestyle='dashed',
          linewidth=1, markersize=5)

ax4.plot(x,Data["Türk_Farklı_Sayı"],color='red', marker='o', linestyle='dashed',
          linewidth=1, markersize=5)

ax3.set_xticklabels(Data.index, rotation=90)
ax3.set_xticks(x)

# Customize axes scales
ax3.set_ylim(0, 5e8)
ax4.set_ylim(0, 7500)
# Combine the legends

h1, l1 = ax3.get_legend_handles_labels()  ### h1 göstergesi l1 label adı
h2, l2 = ax4.get_legend_handles_labels()  ### h2 göstergesi l2 label adı
ax3.legend(h1+h2, l1+l2)

ax3.set(xlabel='Ports',ylabel='Total Output-Ton')
ax4.set(ylabel='Different Ship Call')
ax3.set_title("Port Output Relation with Number of Different Ship Calls (2011-2019)",
              {'fontsize':18,"fontstyle":"italic", "fontweight":"bold"})




Yabancı_Toplam=Merged_Data1.groupby("Liman")["Toplam_Gemi_Yabancı_Sayı"].sum()
Port_Data=Merged_Data1.groupby("Liman")["Toplam_elleçleme"].sum()
Turk_Toplam=Merged_Data1.groupby("Liman")["Toplam_Türk_Sayı"].sum()

Data4=pd.DataFrame(Yabancı_Toplam)
Data4["Toplam_elleçleme"]=Port_Data
Data4["Toplam_Türk_Sayı"]=Turk_Toplam
Data4=Data4.sort_values(by="Toplam_elleçleme",ascending=False)


Data4.describe()

plt.style.use('seaborn-white')
Data=Data4
x=range(53*150)[::150]

colors = sns.color_palette()
plt.figure(figsize=(12,8))

# Setup the dual y-axes
ax3 = plt.axes()
ax4 = ax3.twinx()   ## We say use same x but different y scale...!!!!!!


# Plot the linear regression data
ax3.bar(x,Data["Toplam_elleçleme"], width=70, edgecolor="black", facecolor="wheat",
       align="center")
#plt.xticks(x,label=Data.index,rotation='vertical')


# Plot the regularization data sets
ax4.plot(x,Data["Toplam_Gemi_Yabancı_Sayı"],color='green', marker='o', linestyle='dashed',
          linewidth=1, markersize=5)

ax4.plot(x,Data["Toplam_Türk_Sayı"],color='red', marker='o', linestyle='dashed',
          linewidth=1, markersize=5)

ax3.set_xticklabels(Data.index, rotation=90)
ax3.set_xticks(x)

# Customize axes scales
ax3.set_ylim(0, 5e8)
ax4.set_ylim(0, 30000)
# Combine the legends

h1, l1 = ax3.get_legend_handles_labels()  ### h1 göstergesi l1 label adı
h2, l2 = ax4.get_legend_handles_labels()  ### h2 göstergesi l2 label adı
ax3.legend(h1+h2, l1+l2)

ax3.set(xlabel='Ports',ylabel='Total Output-Ton')
ax4.set(ylabel='Total Ship Call')
ax3.set_title("Port Output Relation with Number of Total Ship Calls (2011-2019)",
              {'fontsize':18,"fontstyle":"italic", "fontweight":"bold"})



Toplam_Yıl=Merged_Data1.groupby("Yıl")["Toplam_elleçleme"].sum()
Toplam_Gemi_Yabancı_Sayı=Merged_Data1.groupby("Yıl")["Toplam_Gemi_Yabancı_Sayı"].sum()
Toplam_Türk_Sayı=Merged_Data1.groupby("Yıl")["Toplam_Türk_Sayı"].sum()
Data5=pd.DataFrame(Toplam_Yıl)
Data5["Toplam_Gemi_Yabancı_Sayı"]=Toplam_Gemi_Yabancı_Sayı
Data5["Toplam_Türk_Sayı"]=Toplam_Türk_Sayı
Data5.describe()
plt.style.use('seaborn-white')
Data=Data5
x=range(8*150)[::150]

colors = sns.color_palette()
plt.figure(figsize=(12,8))

# Setup the dual y-axes
ax3 = plt.axes()
ax4 = ax3.twinx()   ## We say use same x but different y scale...!!!!!!


# Plot the linear regression data
ax3.bar(x,Data["Toplam_elleçleme"], width=70, edgecolor="black", facecolor="wheat",
       align="center")
#plt.xticks(x,label=Data.index,rotation='vertical')


# Plot the regularization data sets
ax4.plot(x,Data["Toplam_Gemi_Yabancı_Sayı"],color='green', marker='o', linestyle='dashed',
          linewidth=2, markersize=8)

ax4.plot(x,Data["Toplam_Türk_Sayı"],color='red', marker='o', linestyle='dashed',
          linewidth=2, markersize=8)

ax3.set_xticklabels(Data.index, rotation=90)
ax3.set_xticks(x)

# Customize axes scales
ax3.set_ylim(0, 5e8)
ax4.set_ylim(0, 50000)
# Combine the legends

h1, l1 = ax3.get_legend_handles_labels()  ### h1 göstergesi l1 label adı
h2, l2 = ax4.get_legend_handles_labels()  ### h2 göstergesi l2 label adı
ax3.legend(h1+h2, l1+l2)

ax3.set(xlabel='Year',ylabel='Total Output-Ton')
ax4.set(ylabel='Total Ship Call')
ax3.set_title("Port Output Relation with Number of Total Ship Calls (2012-2019) by Year",
              {'fontsize':18,"fontstyle":"italic", "fontweight":"bold"})



