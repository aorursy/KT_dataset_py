
%%capture
!pip install sweetviz
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
#import sweetviz
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
data4IBM=pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv", header=0)
data4IBM.head(10)
data4IBM.describe()
data4IBM.info()
sweetreport = sweetviz.analyze(data4IBM, target_feat="Attrition")
sweetreport.show_html('sweetreport.html')
corr4=data4IBM.corr()
corr4
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(corr4, annot=True, linewidths=.5, fmt= '.1f',ax=ax)

deep_dive1 = ['Age','Over18','Gender']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,5))
fig.subplots_adjust(hspace=0.5, bottom=0)
for ax, catplot in zip(axes.flatten(), deep_dive1):
        sns.countplot(x=catplot, data=data4IBM, hue='Attrition',  palette="Set3", ax=ax, )
        ax.set_title(catplot.upper(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel(f'{catplot} Values', fontsize=12)
        ax.legend(title='Attrition', fontsize=12)
deep_dive2 = ['EducationField','Education','Department']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,5))
fig.subplots_adjust(hspace=0.5, bottom=0)
for ax, catplot in zip(axes.flatten(), deep_dive2):
        sns.countplot(x=catplot, data=data4IBM, hue='Attrition',  palette="Set3", ax=ax, )
        ax.set_title(catplot.upper(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel(f'{catplot} Values', fontsize=12)
        ax.legend(title='Attrition', fontsize=12)
deep_dive3 = ['BusinessTravel', 'DistanceFromHome', 'EnvironmentSatisfaction']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,5))
fig.subplots_adjust(hspace=0.5, bottom=0)
for ax, catplot in zip(axes.flatten(), deep_dive3):
        sns.countplot(x=catplot, data=data4IBM, hue='Attrition',  palette="Set3", ax=ax, )
        ax.set_title(catplot.upper(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel(f'{catplot} Values', fontsize=12)
        ax.legend(title='Attrition', fontsize=12)
        ax.legend(title='Attrition', fontsize=12)
        
deep_dive4 = ['MaritalStatus','WorkLifeBalance','OverTime']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,5))
fig.subplots_adjust(hspace=0.5, bottom=0)
for ax, catplot in zip(axes.flatten(), deep_dive4):
        sns.countplot(x=catplot, data=data4IBM, hue='Attrition', palette="Set3", ax=ax, )
        ax.set_title(catplot.upper(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel(f'{catplot} Values', fontsize=12)
        ax.legend(title='Attrition', fontsize=12)

deep_dive7 = ['StandardHours','RelationshipSatisfaction','TrainingTimesLastYear']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,5))
fig.subplots_adjust(hspace=0.5, bottom=0)
for ax, catplot in zip(axes.flatten(), deep_dive7):
        sns.countplot(x=catplot, data=data4IBM, hue='Attrition',  palette="Set3", ax=ax, )
        ax.set_title(catplot.upper(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel(f'{catplot} Values', fontsize=12)
        ax.legend(title='Attrition', fontsize=12)
deep_dive5 = ['JobLevel', 'JobInvolvement', 'JobSatisfaction'] 

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,5))
fig.subplots_adjust(hspace=0.5, bottom=0)
for ax, catplot in zip(axes.flatten(), deep_dive5):
        sns.countplot(x=catplot, data=data4IBM, hue='Attrition',  palette="Set3", ax=ax, )
        ax.set_title(catplot.upper(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel(f'{catplot} Values', fontsize=12)
        ax.legend(title='Attrition', fontsize=12)
deep_dive6 = ['TotalWorkingYears', 'NumCompaniesWorked', 'YearsAtCompany']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,5))
fig.subplots_adjust(hspace=0.5, bottom=0)
for ax, catplot in zip(axes.flatten(), deep_dive6):
        sns.countplot(x=catplot, data=data4IBM, hue='Attrition',  palette="Set3", ax=ax, )
        ax.set_title(catplot.upper(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel(f'{catplot} Values', fontsize=12)
        ax.legend(title='Attrition', fontsize=12)
deep_dive7 = ['YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,5))
fig.subplots_adjust(hspace=0.5, bottom=0)
for ax, catplot in zip(axes.flatten(), deep_dive7):
        sns.countplot(x=catplot, data=data4IBM, hue='Attrition',  palette="Set3", ax=ax, )
        ax.set_title(catplot.upper(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel(f'{catplot} Values', fontsize=12)
        ax.legend(title='Attrition', fontsize=12) 
deep_dive7 = ['MonthlyIncome', 'PercentSalaryHike', 'PerformanceRating']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,5))
fig.subplots_adjust(hspace=0.5, bottom=0)
for ax, catplot in zip(axes.flatten(), deep_dive7):
        sns.countplot(x=catplot, data=data4IBM, hue='Attrition',  palette="Set3", ax=ax, )
        ax.set_title(catplot.upper(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel(f'{catplot} Values', fontsize=12)
        ax.legend(title='Attrition', fontsize=12) 
data16 = pd.read_csv("/kaggle/input/cattells-16-personality-factors/16PF/data.csv", sep="\t")
data16.head(10)
data16.info()
data16.describe()
corr1=data16.corr()

corr1.style.background_gradient(cmap='coolwarm')

corr1 

#corr1.to_csv("D:\corr1.csv")
f,ax = plt.subplots(figsize=(50,50))
sns.heatmap(corr1, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
data16['A'] = (data16['A1']+data16['A2']+data16['A3']+data16['A4']+data16['A5']+data16['A6']+data16['A7']+data16['A8']+data16['A9']+data16['A10'])/10
data16['B'] = (data16['B1']+data16['B2']+data16['B3']+data16['B4']+data16['B5']+data16['B6']+data16['B7']+data16['B8']+data16['B9']+data16['B10']+data16['B11']+data16['B12']+data16['B13'])/13
data16['C'] = (data16['C1']+data16['C2']+data16['C3']+data16['C4']+data16['C5']+data16['C6']+data16['C7']+data16['C8']+data16['C9']+data16['C10'])/10
data16['D'] = (data16['D1']+data16['D2']+data16['D3']+data16['D4']+data16['D5']+data16['D6']+data16['D7']+data16['D8']+data16['D9']+data16['D10'])/10
data16['E'] = (data16['E1']+data16['E2']+data16['E3']+data16['E4']+data16['E5']+data16['E6']+data16['E7']+data16['E8']+data16['E9']+data16['E10'])/10
data16['F'] = (data16['F1']+data16['F2']+data16['F3']+data16['F4']+data16['F5']+data16['F6']+data16['F7']+data16['F8']+data16['F9']+data16['F10'])/10
data16['G'] = (data16['G1']+data16['G2']+data16['G3']+data16['G4']+data16['G5']+data16['G6']+data16['G7']+data16['G8']+data16['G9']+data16['G10'])/10
data16['H'] = (data16['H1']+data16['H2']+data16['H3']+data16['H4']+data16['H5']+data16['H6']+data16['H7']+data16['H8']+data16['H9']+data16['H10'])/10
data16['I'] = (data16['I1']+data16['I2']+data16['I3']+data16['I4']+data16['I5']+data16['I6']+data16['I7']+data16['I8']+data16['I9']+data16['I10'])/10
data16['J'] = (data16['J1']+data16['J2']+data16['J3']+data16['J4']+data16['J5']+data16['J6']+data16['J7']+data16['J8']+data16['J9']+data16['J10'])/10
data16['K'] = (data16['K1']+data16['K2']+data16['K3']+data16['K4']+data16['K5']+data16['K6']+data16['K7']+data16['K8']+data16['K9']+data16['K10'])/10
data16['L'] = (data16['L1']+data16['L2']+data16['L3']+data16['L4']+data16['L5']+data16['L6']+data16['L7']+data16['L8']+data16['L9']+data16['L10'])/10
data16['M'] = (data16['M1']+data16['M2']+data16['M3']+data16['M4']+data16['M5']+data16['M6']+data16['M7']+data16['M8']+data16['M9']+data16['M10'])/10
data16['N'] = (data16['N1']+data16['N2']+data16['N3']+data16['N4']+data16['N5']+data16['N6']+data16['N7']+data16['N8']+data16['N9']+data16['N10'])/10
data16['O'] = (data16['O1']+data16['O2']+data16['O3']+data16['O4']+data16['O5']+data16['O6']+data16['O7']+data16['O8']+data16['O9']+data16['O10'])/10
data16['P'] = (data16['P1']+data16['P2']+data16['P3']+data16['P4']+data16['P5']+data16['P6']+data16['P7']+data16['P8']+data16['P9']+data16['P10'])/10
data16.drop(columns = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11","B12","B13","C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"], inplace=True) 
data16
data16.describe()
corr16short=data16.corr()

corr16short
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(corr16short, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
corr16short.style.background_gradient(cmap='coolwarm')
data5 = pd.read_csv("../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv", sep='\t')
data5.head()
data5.describe()
data5.info()
data5.AGR7.value_counts(normalize=False).plot.bar()
plt.show
data5.AGR7.value_counts(normalize=False, ascending=False)
data5.CSN4.value_counts(normalize=False).plot.bar()
plt.show
data5.CSN4.value_counts(normalize=False, ascending=False)
data5.CSN8.value_counts(normalize=False).plot.bar()
plt.show
data5.CSN8.value_counts(normalize=False, ascending=False)
data5.EXT4.value_counts(normalize=False).plot.bar()
plt.show
data5.EXT4.value_counts(normalize=False, ascending=False)
data5.EST1.value_counts(normalize=False).plot.bar()
plt.show
data5.EST1.value_counts(normalize=False, ascending=False)
data5.CSN1.value_counts(normalize=False).plot.bar()
plt.show
data5.CSN1.value_counts(normalize=False, ascending=False)
data5.AGR2.value_counts(normalize=False).plot.bar()
plt.show
data5.AGR2.value_counts(normalize=False, ascending=False)
data5.CSN7.value_counts(normalize=False).plot.bar()
plt.show
data5.CSN7.value_counts(normalize=False, ascending=False)
data5.CSN9.value_counts(normalize=False).plot.bar()
plt.show
data5.CSN9.value_counts(normalize=False, ascending=False)
data5.CSN10.value_counts(normalize=False).plot.bar()
plt.show
data5.CSN10.value_counts(normalize=False, ascending=False)
data5.CSN4.value_counts(normalize=False).plot.bar()
plt.show
data5.CSN4.value_counts(normalize=False, ascending=False)
data5.CSN2.value_counts(normalize=False).plot.bar()
plt.show
data5.CSN2.value_counts(normalize=False, ascending=False)
data5.EST5.value_counts(normalize=False).plot.bar()
plt.show
data5.EST5.value_counts(normalize=False, ascending=False)
data5.AGR5.value_counts(normalize=False).plot.bar()
plt.show
data5.AGR5.value_counts(normalize=False, ascending=False)
data5.AGR1.value_counts(normalize=False).plot.bar()
plt.show
data5.AGR1.value_counts(normalize=False, ascending=False)
data5.EXT8.value_counts(normalize=False).plot.bar()
plt.show
data5.EXT8.value_counts(normalize=False, ascending=False)
data5.CSN3.value_counts(normalize=False).plot.bar()
plt.show
data5.CSN3.value_counts(normalize=False, ascending=False)
data5.OPN7.value_counts(normalize=False).plot.bar()
plt.show
data5.OPN7.value_counts(normalize=False, ascending=False)
data5.CSN7.value_counts(normalize=False).plot.bar()
plt.show
data5.CSN7.value_counts(normalize=False, ascending=False)
data5.CSN9.value_counts(normalize=False).plot.bar()
plt.show
data5.CSN9.value_counts(normalize=False, ascending=False)
data5.EXT9.value_counts(normalize=False).plot.bar()
plt.show
data5.EXT9.value_counts(normalize=False, ascending=False)
data5.EST10.value_counts(normalize=False).plot.bar()
plt.show
data5.EST10.value_counts(normalize=False, ascending=False)
data5.AGR5.value_counts(normalize=False).plot.bar()
plt.show
data5.AGR5.value_counts(normalize=False, ascending=False)
data5.CSN8.value_counts(normalize=False).plot.bar()
plt.show
data5.CSN8.value_counts(normalize=False, ascending=False)
data5.AGR3.value_counts(normalize=False).plot.bar()
plt.show
data5.AGR3.value_counts(normalize=False, ascending=False)
data5.EXT2.value_counts(normalize=False).plot.bar()
plt.show
data5.EXT2.value_counts(normalize=False, ascending=False)
data5.EST3.value_counts(normalize=False).plot.bar()
plt.show
data5.EST3.value_counts(normalize=False, ascending=False)
data5.CSN6.value_counts(normalize=False).plot.bar()
plt.show
data5.CSN6.value_counts(normalize=False, ascending=False)
data5.EST9.value_counts(normalize=False).plot.bar()
plt.show
data5.EST9.value_counts(normalize=False, ascending=False)
data5.OPN1.value_counts(normalize=False).plot.bar()
plt.show
data5.OPN1.value_counts(normalize=False, ascending=False)