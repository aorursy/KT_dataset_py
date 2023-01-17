import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/haberman.csv')#haberman data is loaded into dataframe

df.columns=['age','op_year','axil','survival']#giving names to columns
df.head()#data displayonly first 5

df.shape #size of data
#changing the survival status from 1 and to 2 as yes and no
df['survival']=df['survival'].map({1:'Alive',2:'Died'})
df['survival']=df['survival'].astype('category')
df['survival'].value_counts()#counts data points for each class
df.info()#no,of data points for each variable
#Statstical observations
df.describe()
#univariate
sns.FacetGrid(df, hue="survival", height=5).map(sns.distplot, "age").add_legend();
plt.show()
sns.FacetGrid(df, hue="survival", height=5).map(sns.distplot, "op_year").add_legend();
plt.show()
sns.FacetGrid(df, hue="survival", height=5).map(sns.distplot, "axil").add_legend();
plt.show()
#CDF/PDF
#PDF/CDF of age 
df_alive=df.loc[df['survival']=="Alive"]
df_died=df.loc[df['survival']=="Died"]
counts,bin_edges=np.histogram(df_alive['age'],bins=10,density=True)
pdf=counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='alive_pdf')
plt.plot(bin_edges[1:],cdf,label='alive_cdf')


counts,bin_edges=np.histogram(df_died['age'],bins=10,density=True)
pdf=counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label="died_pdf")
plt.plot(bin_edges[1:],cdf,label="died_cdf")
plt.xlabel('age')
plt.ylabel('PDF/CDF')
plt.title('Pdf/Cdf for ages')
plt.legend()
plt.show()


#CDF/PDF of axil nodes
counts,bin_edges=np.histogram(df_alive['axil'],bins=10,density=True)
pdf=counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='alive_pdf')
plt.plot(bin_edges[1:],cdf,label='alive_cdf')

counts,bin_edges=np.histogram(df_died['axil'],bins=10,density=True)
pdf=counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label="died_pdf")
plt.plot(bin_edges[1:],cdf,label="died_cdf")
plt.xlabel('axil')
plt.ylabel('PDF/CDF')
plt.title('Pdf/Cdf for axil nodes')
plt.legend()
plt.show()


#CDF/PDF of year of operation
counts,bin_edges=np.histogram(df_alive['op_year'],bins=10,density=True)
pdf=counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='alive_pdf')
plt.plot(bin_edges[1:],cdf,label='alive_cdf')

counts,bin_edges=np.histogram(df_died['op_year'],bins=10,density=True)
pdf=counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label="died_pdf")
plt.plot(bin_edges[1:],cdf,label="died_cdf")
plt.xlabel('op_year')
plt.ylabel('PDF/CDF')
plt.title('Pdf/Cdf for operation year')
plt.legend()

plt.show()






# Box Plots 
sns.boxplot(x="survival",y="age",data=df)
plt.title('box plot of age')
plt.show()
sns.boxplot(x="survival",y="op_year",data=df)
plt.title('boxplot of op_year')
plt.show()
sns.boxplot(x="survival",y="axil",data=df)
plt.title('box plot of axil')
plt.show()
# voilin Plots 
sns.violinplot(x="survival",y="age",data=df)
plt.title('violinplot for age')
plt.show()
sns.violinplot(x="survival",y="op_year",data=df)
plt.title('violinplot for op_year')
plt.show()
sns.violinplot(x="survival",y="axil",data=df)
plt.title('violinplot for axil')
plt.show()
#Violin plots are used here to know the disturbution of data
#multivariate analysis
sns.set_style('whitegrid')
sns.pairplot(df,hue='survival',height=6).add_legend()
plt.show()