import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


df_input = pd.read_csv("../input/haberman.csv")
df_input.shape
columns =["Age","Year_of_Operation","positive_axillary_node","survival_status"]
df_input.columns = columns
df = df_input.copy()
df_input.head()
mask = (df.survival_status == 1)
df.loc[mask,"survival_status"] = "Survived more than 5 years"
df.loc[~mask,"survival_status"] = "Survived less than 5 years"
df_input.head()
df.describe()
df.isnull().sum()

print("Number of Rows in dataset" ,df.shape[0])
print("Number of columns in dataset" ,df.shape[1])
df_not_survived = df[df.survival_status == "Survived less than 5 years"]
ptnt_died_within5yrs = df_not_survived.shape[0]
df_survived = df[df.survival_status == "Survived more than 5 years"]
patient_survived = df_survived.shape[0]
print(f"{ptnt_died_within5yrs} Patient died within 5 yrs of operation ")
print(f"{patient_survived} patient survived the operation")
counts,bins  = np.histogram(df.positive_axillary_node,bins = 100,density = True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bins[1:],pdf,label = 'PDF')
plt.plot(bins[1:], cdf,label = 'CDF')
plt.title("positive_axillary_node")
plt.legend()

counts,bins  = np.histogram(df.Age,bins = 50,density = True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bins[1:],pdf,label = 'PDF')
plt.plot(bins[1:], cdf,label = 'CDF')
plt.title("Age")
plt.legend()
counts,bins  = np.histogram(df.Year_of_Operation,bins = 50,density = True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bins[1:],pdf, label='PDF')
plt.plot(bins[1:], cdf,label = 'CDF')
plt.title("Year_of_Operation")
plt.legend()


df_not_survived.Age.mean()
df_survived.Age.mean()

sns.FacetGrid(df,hue = "survival_status",height = 4,aspect = 4).map(sns.distplot,"Age",bins =100).add_legend()
sns.FacetGrid(df,hue = "survival_status",height = 4,aspect = 4).map(sns.distplot,"positive_axillary_node" ,bins =200).add_legend()
sns.FacetGrid(df,hue = "survival_status",height = 4,aspect = 4).map(sns.distplot,"Year_of_Operation" ,bins =100).add_legend()

sns.FacetGrid(df,row = "survival_status",hue = 'survival_status' ,height =4,aspect  =4).map(sns.distplot,"Year_of_Operation").add_legend()
sns.FacetGrid(df,row = "survival_status",hue = 'survival_status' ,height =4,aspect  =4).map(sns.distplot,"positive_axillary_node").add_legend()
sns.FacetGrid(df,row = "survival_status",hue = 'survival_status' ,height =4,aspect  =4).map(sns.distplot,"Age").add_legend()    
sns.boxplot(x = "survival_status", y ="Age" , data = df ).set_title("Age")

sns.boxplot(x = "survival_status", y= "positive_axillary_node",data = df).set_title("positive_axillary_node")

sns.boxplot(x = "survival_status", y= "Year_of_Operation",data = df).set_title("Year_of_Operation")
sns.violinplot(x="survival_status",y = "Year_of_Operation",data =df).set_title("Year_of_Operation")
sns.violinplot(x="survival_status",y = "Age",data =df).set_title("Age")
sns.violinplot(x = "survival_status" , y= "positive_axillary_node",data = df).set_title("positive_axillary_node")

sns.set_style("darkgrid");
sns.pairplot(df_input,hue="survival_status",height=5)