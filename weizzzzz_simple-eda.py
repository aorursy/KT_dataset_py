import pandas as pd

df_1 = pd.read_csv("../input/sars-coronavirus-accession/MN997409.1-4NY0T82X016-Alignment-HitTable.csv")

df_2 = pd.read_csv("../input/sars-coronavirus-accession/SARS_CORONAVIRUS_287BP_MN975263.1_accession_nucleotide.csv")

df_3 = pd.read_csv("../input/sars-coronavirus-accession/SARS_CORONAVIRUS_287BP_MN975263.1_accession_protein.csv")
df_1.shape
df_1.head()

df_1.columns=['column1','column2','column3','column4','column5','column6','column7','column8','column9','column10','column11','column12']
df_1.isnull().sum()
df_1.dtypes
import seaborn as sns

sns.heatmap(df_1.corr())
sns.pairplot(df_1)
df_2.head()
df_2.shape
df_2.isnull().sum()
df_21=df_2.drop(['Genotype','Genome_Region','Segment','Details','BioSample','Nuc._Completeness'],axis=1)
df_22=df_21.dropna()
df_22.shape
df_22
df_22.columns
sns.barplot(df_22.groupby('Geo_Location').size().index,df_22.groupby('Geo_Location').size())

sns.barplot(df_22.groupby('Isolation_Source').size().index,df_22.groupby('Isolation_Source').size())
df_3.isnull().sum()
df_31=df_3.drop(['Details','Genotype','Genome_Region','Segment','BioSample','Isolation_Source'],axis=1)
df_31.isnull().sum()