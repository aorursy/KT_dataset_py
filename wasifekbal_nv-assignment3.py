import pandas as pd
import numpy as np
import math
import seaborn as sns
df=pd.read_csv('../input/iris/Iris.csv')
## changing the name of the columns.....
df.rename(columns={'SepalLengthCm':'sepal_length','SepalWidthCm':'sepal_width','PetalLengthCm':'petal_length','PetalWidthCm':'petal_width','Species':'species'},inplace=True)
a = sns.FacetGrid(df, col="species")
b = sns.FacetGrid(df, col="species")
c = sns.FacetGrid(df, col="species")
d = sns.FacetGrid(df, col="species")
a.map(sns.distplot,"sepal_length")
b.map(sns.distplot,"sepal_width")
c.map(sns.distplot,"petal_length")
d.map(sns.distplot,"petal_width")
## creating separate dataframes for the species.....
setosa=df[df['species']=='Iris-setosa']
versicolor=df[df['species']=='Iris-versicolor']
virginica=df[df['species']=='Iris-virginica']
## Calculating the mean and standared deviation of all the columns for all the species separately.

# setosa
mean_sl_set=setosa['sepal_length'].mean()
sd_sl_set=setosa['sepal_length'].std()

mean_sw_set=setosa['sepal_width'].mean()
sd_sw_set=setosa['sepal_width'].std()

mean_pl_set=setosa['petal_length'].mean()
sd_pl_set=setosa['petal_length'].std()

mean_pw_set=setosa['petal_width'].mean()
sd_pw_set=setosa['petal_width'].std()

# versicolor
mean_sl_ver=versicolor['sepal_length'].mean()
sd_sl_ver=versicolor['sepal_length'].std()

mean_sw_ver=versicolor['sepal_width'].mean()
sd_sw_ver=versicolor['sepal_width'].std()

mean_pl_ver=versicolor['petal_length'].mean()
sd_pl_ver=versicolor['petal_length'].std()

mean_pw_ver=versicolor['petal_width'].mean()
sd_pw_ver=versicolor['petal_width'].std()

# virginica
mean_sl_vir=virginica['sepal_length'].mean()
sd_sl_vir=virginica['sepal_length'].std()

mean_sw_vir=virginica['sepal_width'].mean()
sd_sw_vir=virginica['sepal_width'].std()

mean_pl_vir=virginica['petal_length'].mean()
sd_pl_vir=virginica['petal_length'].std()

mean_pw_vir=virginica['petal_width'].mean()
sd_pw_vir=virginica['petal_width'].std()
# normal distribution function.....
def ndf(x,mean,sd):
    return ((math.exp((-((x*x)+(mean*mean)-(2*x*mean)))/2*sd*sd))/(sd*1.414*3.14))
#((math.exp((-(x**2)-(mean**2)+(2*x*mean))/2*sd*sd)/sd)/sd*1.414*3.14)
#probability of setosa
p_setosa = ndf(4.7,mean_sl_set,sd_sl_set)*ndf(3.7,mean_sw_set,sd_sw_set)*ndf(2,mean_pl_set,sd_pl_set)*ndf(0.3,mean_pw_set,sd_pw_set)*(1/3)

# probability of versicolor
p_versicolor = ndf(4.7,mean_sl_ver,sd_sl_ver)*ndf(3.7,mean_sw_ver,sd_sw_ver)*ndf(2,mean_pl_ver,sd_pl_ver)*ndf(0.3,mean_pw_ver,sd_pw_ver)*(1/3)

# probability of virginica
p_virginica = p_setosa = ndf(4.7,mean_sl_vir,sd_sl_vir)*ndf(3.7,mean_sw_vir,sd_sw_vir)*ndf(2,mean_pl_vir,sd_pl_vir)*ndf(0.3,mean_pw_vir,sd_pw_vir)*(1/3)
print(p_setosa)
print(p_versicolor)
print(p_virginica)
maxx = max(p_setosa,p_versicolor,p_virginica)
if p_setosa==maxx:
    print("It's Setosa.")
elif p_versicolor==maxx:
    print("It's Versicolor.")
else:
    print("It's Virginica.")