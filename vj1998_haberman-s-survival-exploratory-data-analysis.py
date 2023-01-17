# Importing libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading data
df = pd.read_csv("../input/haberman.csv", names=['Age', 'Year', 'Axillary nodes dect', 'Survstatus'])
df
print(df.shape) # output indicates (rows, columns)
print(df.columns) # output gives columns name which have assigned and its type
# We have two classes in Survstatus:
# person who is survived after the surgery.
# person is dead after the surgery.

df['Survstatus'].value_counts()
# For Age and Axillary nodes dect
sns.set_style('whitegrid')
sns.FacetGrid(df, hue='Survstatus', height=3) \
    .map(plt.scatter, 'Age', 'Axillary nodes dect') \
    .add_legend()
plt.show()
# For Age and Year

sns.set_style('whitegrid')
sns.FacetGrid(df, hue="Survstatus", height=3) \
    .map(plt.scatter, 'Age', 'Year') \
    .add_legend()
plt.show()
# For Year and Axillary nodes dect
sns.set_style('whitegrid')
sns.FacetGrid(df, hue='Survstatus', height=3) \
    .map(plt.scatter, 'Year', 'Axillary nodes dect') \
    .add_legend()
plt.show()
sns.set_style('whitegrid')
sns.pairplot(df, hue='Survstatus', vars=['Age', 'Year', 'Axillary nodes dect'], height=3)
plt.show()
# plotting one dimensionally
sns.FacetGrid(df, hue='Survstatus', height=5) \
    .map(sns.distplot, 'Age') \
    .add_legend()
sns.FacetGrid(df, hue='Survstatus', size=5) \
    .map(sns.distplot, 'Axillary nodes dect') \
    .add_legend()
sns.FacetGrid(df, hue='Survstatus', height=5) \
    .map(sns.distplot, 'Year') \
    .add_legend()
## Analysing more on yearly basis using histogram, PDF and CDF
df_sur = df.loc[df["Survstatus"]==1]
counts, bin_edges = np.histogram(df_sur['Axillary nodes dect'], bins=10, density=True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend('survivalstatus')
plt.legend(['Survived_PDF', 'Survived_CDF'])
plt.xlabel("Age of Survived")
plt.show()
df_dead = df.loc[df['Survstatus']==2]
counts, bin_edges = np.histogram(df_dead['Axillary nodes dect'], bins=10, density=True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend('survivalstatus')
plt.legend(['Survived_PDF', 'Survived_CDF'])
plt.xlabel("Age of dead")
plt.show()
counts, bin_edges = np.histogram(df_dead['Axillary nodes dect'], bins=10, density=True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(df_sur['Axillary nodes dect'], bins=10, density=True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend('survstatus')
plt.legend(['Survived_PDF', 'Survived_CDF','Died_PDF', 'Died_CDF'])
plt.xlabel("Age of dead")
plt.show()
sns.boxplot(x='Survstatus', y='Axillary nodes dect', data=df)
plt.show()
sns.violinplot(x='Survstatus', y='Axillary nodes dect', data=df, size=5)
plt.show()
#2D Density plot, contors-plot
sns.jointplot(x="Survstatus", y="Axillary nodes dect", data=df, kind="kde");
plt.show();