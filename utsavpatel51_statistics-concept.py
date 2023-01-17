import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from statsmodels.graphics.gofplots import qqplot

import seaborn as sns

from scipy.stats import binom

from scipy.stats import bernoulli

from scipy.stats import norm

from scipy.stats import uniform

from scipy.stats import poisson

from scipy.stats import sem

import scipy
#genrate test dataset

np.random.seed(1)

data =  5 * np.random.randn(100) + 50

np.mean(data) , np.std(data)
#HIST PLOT

plt.hist(data)

plt.show()
#QQ PLOT

#https://www.youtube.com/watch?v=okjYjClSjOg

qqplot(data,line='s')#line='s' means standarrized line to match

plt.show()
# H0 - sample was drawn from a Gaussian distribution

# H1 - sample was not 

# p <= alpha: reject H0, not normal.

# p > alpha: fail to reject H0, normal.
#Shapiro-Wilk Test

    #suitable for smaller samples of data, e.g. thousands of observations or fewer

#TEST STATASTIC -> how far a sample estimate is from what we expected if H0 is true

from scipy.stats import shapiro



test_stat , p = shapiro(data)

print("test staistic",test_stat),print("p_value",p)

alpha = 0.05

if p <=alpha:print("reject H0")

else:print("fail to reject H0")
#D’Agostino’s K^2 Test

from scipy.stats import normaltest

test_stat , p = normaltest(data) #test_stat = s^2 + k^2 where s=skew and k=Kurtosis

print("test staistic",test_stat),print("p_value",p)

alpha = 0.05

if p <=alpha:print("reject H0")

else:print("fail to reject H0")
#Anderson-Darling Test

from scipy.stats import anderson

result = anderson(data,dist='norm')

print("test statistic",result.statistic)

print("sl","  | ","cv")

for i in range(len(result.critical_values)):

    sl,cv = result.significance_level[i],result.critical_values[i]

    if result.statistic < cv:

        print(sl,"|",cv,"fail to reject H0")

    else:

        print(sl,cv,"reject H0")
#PMF(discrete variable)

n=10

p=0.5

#X = np.arange(binom.ppf(0.01,n,p),binom.ppf(0.99,n,p))

X=range(11)

plt.plot(X,binom.pmf(X, n, p),'bo',ms=8,label='binom pmf')

plt.vlines(X,0,binom.pmf(X,n,p),colors='b', lw=5, alpha=0.5)

plt.legend()

plt.show()
#x = np.arange(norm.ppf(0.01),norm.ppf(0.99),0.01)

x = np.linspace(3,-3,100)

plt.plot(x,norm.pdf(x),'r-')

r = norm.rvs(size=1000)

plt.hist(r,density=True,histtype='stepfilled', alpha=0.2)

plt.show()
# For one reason because we made the assumption that the [-3,3] interval fits well to describe

#the normal distribution (and we were right!).But what if we don't really know 

#the practical limits of the x values for a specific distribution

#for any distribution the ppf() function returns a x value that corresponds to the probability that this value appears.

rv = norm(loc=0, scale=1)           # Freeze the norm distribution at mean 0 and std 1

x = np.linspace(rv.ppf(0.0001), rv.ppf(0.9999), 100) #give a value of x where cdf value is 0.0001 and 0.9999

y = norm.pdf(x)

plt.plot(x,y)

plt.show()
#CDF

std=2

mean=0

x = np.linspace(-2*std+mean,2*std+mean,100)

y = norm.pdf(x,loc=mean,scale=std)

plt.plot(x, y, label='PDF')

plt.plot(x,norm.cdf(x), 'r', label='CDF')

plt.legend(loc='best')

plt.show()
#UNIFORM DISTRIBUTION

dist = np.random.random(1000)

sns.scatterplot(data=dist)

plt.title('Scatterplot for uniform distribution')
#plt.hist(dist)

sns.distplot(dist)

plt.title('uniform distribution')
#Bernoulli distribution

#head 1 tail 0

p=0.6 #head with prob p=0.6

b_dist = bernoulli.rvs(p,size=1000)

df = pd.DataFrame({'value':b_dist})

df[df==0].count().value,df[df==1].count().value#no of head and no of tail
df_count = df['value'].value_counts()

df_count.plot(kind='bar', rot=0)
#UNIFORM

uniform_d = uniform.rvs(size=100,loc=0,scale=10)

sns.distplot(uniform_d,bins=10,kde=True)

plt.xlabel('Uniform')

plt.ylabel('Frequency')
#NORMAL

norm_d = norm.rvs(size=10000,loc=0,scale=2)

sns.distplot(norm_d,bins=50,kde=True)

plt.xlabel('Normal')

plt.ylabel('Frequency')
#Bernouli

data_bern = bernoulli.rvs(size=10000,p=0.6)

ax = sns.distplot(data_bern,kde=False)

plt.xlabel('bernoulli')

plt.ylabel('Frequency')
#Binomial

binom_d = binom.rvs(n=10,p=0.5,size=1000)

sns.distplot(binom_d,kde=False,color='blue')

plt.xlabel('Binomal')

plt.ylabel('Frequency')
#poisson

poisson_d = poisson.rvs(mu=3,size=10000)

sns.distplot(poisson_d,kde=False,label='rate=3')

plt.xlabel('Possion')

plt.ylabel('Frequency')

plt.legend()
norm_data = pd.DataFrame({'Value':norm_d})

print(norm_data.describe())

print()

print('Mean',norm_d.mean())

print('var',norm_d.var())

print('std',norm_d.std())

print('Skewness',norm_data['Value'].skew())

print('Kurtosis',norm_data['Value'].kurtosis())
print('Standard error of uniform sample',sem(uniform_d))

print('Standard error of norm sample',sem(norm_d))

print('Standard error of binomial sample',sem(binom_d))
np.random.seed(1)

data1 =  20 * np.random.randn(1000) + 100 #100 mean and 20 std

data2 = data1 + 10 * np.random.randn(1000) + 50

print('data1-> mean: ',data1.mean(),',std: ',data1.std())

print('data2-> mean: ',data2.mean(),',std: ',data2.std())
sns.scatterplot(data1,data2)
#covariance matrix

#cov(X, Y) = (sum (x - mean(X)) * (y - mean(Y))) * 1/(n-1)

np.cov(data1,data2)

# positive corelate two variable -> 389.7545618

#Problem -> hard to interprate
#Pearson correlation coefficient

#normalization of the covariance between the two variables to give an interpretable score

#Pearson's correlation coefficient = cov(X, Y) / (stdv(X) * stdv(Y))

cor , _ =scipy.stats.pearsonr(data1,data2)

cor
#Spearman’s correlation coefficient

#measure nonlinear relationship and variable may not have gaussian distribution

#Spearman's correlation coefficient = covariance(rank(X), rank(Y)) / (stdv(rank(X)) * stdv(rank(Y)))
np.random.seed(1)

data1 = np.random.rand(1000)*20#not normal distribution

data2 = data1 + (np.random.rand(1000)*10)

#data2 =(np.random.rand(1000)*10) check for both data2 one by one

sns.scatterplot(data1,data2)
# calculate spearman's correlation

coef, p = scipy.stats.spearmanr(data1, data2)

print("coef",coef)



alpha=0.05

if p>alpha:

    print("uncorelated(Failed to reject H0) with confidence",int((1-p)*100))

else:

    print("corelated(reject H0) with confidence",int((1-p)*100))

#Kendall’s correlation

# it calculates a normalized score for the number of matching or concordant rankings between the two samples



coef , p = scipy.stats.kendalltau(data1,data2)

print("coef",coef)



alpha=0.05

if p>alpha:

    print("uncorelated(Failed to reject H0) with confidence",int((1-p)*100))

else:

    print("corelated(reject H0) with confidence",int((1-p)*100))

# p <= alpha: reject null hypothesis, different distribution

# p > alpha: fail to reject null hypothesis, same distribution

# To test we drwan a sample from two diffrent normal distribution
np.random.seed(1)

data1 = 20 * np.random.randn(1000)+50.1

data2 = 10 * np.random.randn(1000)+50

print('Data1 mean is {0} and std is {1}'.format(data1.mean(),data1.std()))

print('Data2 mean is {0} and std is {1}'.format(data2.mean(),data2.std()))
#The Student’s t-test is a statistical hypothesis test that two independent data samples 

#known to have a Gaussian distribution, have the same Gaussian distribution

stat , p = scipy.stats.ttest_ind(data1,data2)

print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha=0.05

if p>alpha:

    print("same distribution(Failed to reject H0) with confidence",int((1-p)*100))

else:

    print("diff distribution(reject H0) with confidence",int((1-p)*100))
np.random.seed(1)

def independent_t_test(data1,data2,alpha=0.05):

    n1=len(data1)

    n2=len(data2)

    mean_data1 = np.mean(data1)

    mean_data2 = np.mean(data2)

    se1=np.std(data1)/(n1**0.5)

    se2=np.std(data2)/(n2**0.5)

    sed=np.sqrt((se1**2)+(se2**2))

    t_stat = (mean_data1-mean_data2)/sed



    degree_fr = n1+n2-2

    critical_value = scipy.stats.t.ppf(1.0-alpha,degree_fr)

    p = (1.0-scipy.stats.t.cdf(abs(t_stat),degree_fr))*2.0

    return t_stat,critical_value,p

data1 = 20 * np.random.randn(1000)+50.1

data2 = 10 * np.random.randn(1000)+50

t_stat , cv , p = independent_t_test(data1,data2)

print('t_stat: ',t_stat,'critical value:',cv,'p value',p)

if p>alpha:

    print("same distribution(Failed to reject H0) with confidence",int((1-p)*100))

else:

    print("diff distribution(reject H0) with confidence",int((1-p)*100))
#H0: Paired sample distributions are equal.

#H1: Paired sample distributions are not equal.

np.random.seed(1)

data1 = 10 * np.random.randn(1000) + 50.055

data2 = 20 * np.random.randn(1000) + 50

stat , p =scipy.stats.ttest_rel(data1,data2)

print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha=0.05

if p>alpha:

    print("same distribution(Failed to reject H0) with confidence",int((1-p)*100))

else:

    print("diff distribution(reject H0) with confidence",int((1-p)*100))
np.random.seed(1)

def dependent_t_test(data1,data2,alpha=0.05):

    n=len(data1)# number of paired samples

    mean_data1 = np.mean(data1)

    mean_data2 = np.mean(data2)

    d1 = sum([(data1[i]-data2[i])**2 for i in range(n)])

    d2 = sum([data1[i]-data2[i] for i in range(n)])

    sd=np.sqrt((d1-(d2**2)/n)/(n-1))

    sed=sd/np.sqrt(n)

    t_stat = (mean_data1-mean_data2)/sed

    degree_fr = n-1

    critical_value = scipy.stats.t.ppf(1.0-alpha,degree_fr)

    p = (1.0-scipy.stats.t.cdf(abs(t_stat),degree_fr))*2.0

    return t_stat,critical_value,p

data1 = 10 * np.random.randn(1000) + 50.055

data2 = 20 * np.random.randn(1000) + 50

alpha=0.05

t_stat,critical_value,p = dependent_t_test(data1,data2,alpha=0.05)

print('t_stat: ',t_stat,'critical value:',cv,'p value',p)

alpha=0.05

if p>alpha:

    print("same distribution(Failed to reject H0) with confidence",int((1-p)*100))

else:

    print("diff distribution(reject H0) with confidence",int((1-p)*100))
data1 = 10 * np.random.randn(1000)+51#this is from diff normal distribution

data2 = 10 * np.random.randn(1000)+50

data3 = 10 * np.random.randn(1000)+50

stat,p = scipy.stats.f_oneway(data1,data2,data3)

print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha=0.05

if p>alpha:

    print("same distribution(Failed to reject H0) with confidence",int((1-p)*100))

else:

    print("one of the distribution differ(reject H0) with confidence",int((1-p)*100))
#Mann-Whitney U Test

#Wilcoxon Signed-Rank Test

#Kruskal-Wallis H Test

#Friedman Test