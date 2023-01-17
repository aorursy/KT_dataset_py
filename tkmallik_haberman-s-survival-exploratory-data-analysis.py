# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
!head ../input/haberman.csv
!wc -l ../input/haberman.csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
raw_data=pd.read_csv("../input/haberman.csv") 
raw_data.head()
raw_data.describe()
raw_data['year'].value_counts().sort_index().plot("line")
plt.title("Number of operations/year")
plt.xlabel("Year")
plt.ylabel("Num of operations")
sns.countplot(x="year",hue='survival_status',data=raw_data).set_title('Survival status of patients/year of operation')
sns.boxplot(x='survival_status',y='Age', data=raw_data).set_title('Box plot for feature AGE')
plt.show()
sns.distplot(raw_data["Age"]).set_title('Histogram for feature AGE')
plt.xlabel("Age")
plt.ylabel("Frequency counts (probability density)")
plt.show()
raw_data.boxplot(column=["positive_axillary_nodes"]).set_title('Box plot for feature axillary nodes')
plt.ylabel("count")
raw_data.hist(column=["positive_axillary_nodes"])
plt.xlabel("Number of axillary nodes ")
plt.ylabel("Count")
plt.title("Histogram of feature num of axillary nodes")
raw_data['positive_axillary_nodes'].value_counts().sort_index()
raw_data['positive_axillary_nodes'].value_counts().sort_index().plot("line")
plt.title("Line chart depicting count of axillary nodes")
plt.xlabel("Num of axillary nodes")
plt.ylabel("Count")
raw_data['survival_status'].value_counts(normalize=True)
raw_data['survival_status'].value_counts().plot("pie")
plt.title("Pie chart depicting the ratio of survival status")
sns.pairplot(raw_data, hue="survival_status",x_vars=['year', 'Age', 'positive_axillary_nodes'],y_vars=['year', 'Age', 'positive_axillary_nodes'], size=3)
plt.show()
sns.FacetGrid(raw_data, hue="survival_status", size=5) \
   .map(sns.distplot, "Age") \
   .add_legend()
plt.title("Histogram of feature age")
plt.ylabel('Frequency counts (probability density)')
plt.show()
sns.FacetGrid(raw_data, hue="survival_status", size=5) \
   .map(sns.distplot, "year") \
   .add_legend()
plt.title("Histogram of feature year_of_operation")
plt.ylabel('Frequency counts (probability density)')
plt.show()
sns.FacetGrid(raw_data, hue="survival_status", size=5) \
   .map(sns.distplot, "positive_axillary_nodes") \
   .add_legend();
plt.title("Histogram of feature num_pos_axillary_nodes")
plt.ylabel('Frequency counts (probability density)')
plt.show();
case_1 = raw_data.loc[raw_data["survival_status"] == 1];
case_2 = raw_data.loc[raw_data["survival_status"] == 2];

#Plot CDF case_1 of age
counts, bin_edges = np.histogram(case_1['Age'],bins=10,density = True)
pdf = counts/(sum(counts))
# compute CDF
cdf = np.cumsum(pdf)
#plt.plot(bin_edges[1:],pdf,label='case_1_age_pdf')
plt.plot(bin_edges[1:],cdf,label='case_1_age_cdf')


#Plot CDF case_2 of age
counts, bin_edges = np.histogram(case_2['Age'],bins=10,density = True)
pdf = counts/(sum(counts))
# compute CDF
cdf = np.cumsum(pdf)
#plt.plot(bin_edges[1:],pdf,label='case_2_age_pdf')
plt.plot(bin_edges[1:],cdf,label='case_2_age_cdf')
plt.ylabel("CDF")
plt.xlabel("Age")
plt.legend()
plt.show()
case_1 = raw_data.loc[raw_data["survival_status"] == 1];
case_2 = raw_data.loc[raw_data["survival_status"] == 2];

#Plot CDF case_1 of num_pos_axillary_nodes
counts, bin_edges = np.histogram(case_1['positive_axillary_nodes'],bins=10,density = True)
pdf = counts/(sum(counts))
# compute CDF
cdf = np.cumsum(pdf)
#plt.plot(bin_edges[1:],pdf,label='case_1_nodes_pdf')
plt.plot(bin_edges[1:],cdf,label='case_1_nodes_cdf')


#Plot CDF case_2 of num_pos_axillary_nodes
counts, bin_edges = np.histogram(case_2['positive_axillary_nodes'],bins=10,density = True)
pdf = counts/(sum(counts))
# compute CDF
cdf = np.cumsum(pdf)
#plt.plot(bin_edges[1:],pdf,label='case_2_nodes_pdf')
plt.plot(bin_edges[1:],cdf,label='case_2_nodes_cdf')
plt.ylabel("CDF")
plt.xlabel("num_pos_axillary_nodes")
plt.legend()
plt.show()
case_1 = raw_data.loc[raw_data["survival_status"] == 1];
case_2 = raw_data.loc[raw_data["survival_status"] == 2];

#Plot CDF case_1 of num_pos_axillary_nodes
counts, bin_edges = np.histogram(case_1['year'],bins=10,density = True)
pdf = counts/(sum(counts))
# compute CDF
cdf = np.cumsum(pdf)
#plt.plot(bin_edges[1:],pdf,label='case_1_oper_year_pdf')
plt.plot(bin_edges[1:],cdf,label='case_1_oper_year_cdf')

#Plot CDF case_2 of num_pos_axillary_nodes
counts, bin_edges = np.histogram(case_2['year'],bins=10,density = True)
pdf = counts/(sum(counts))
# compute CDF
cdf = np.cumsum(pdf)
#plt.plot(bin_edges[1:],pdf,label='case_2_oper_year_pdf')
plt.plot(bin_edges[1:],cdf,label='case_2_oper_year_cdf')
plt.ylabel("CDF")
plt.xlabel("year_of_operation")
plt.legend()
plt.show()
sns.violinplot(x="survival_status", y="Age", inner="quart",data=raw_data, size=8)
plt.title("Violen plot depicting the age distribution")
plt.show()
sns.violinplot(x="survival_status", y="year", inner="quart", data=raw_data, size=8)
plt.title("Violen plot depicting the year_of_operation distribution")
plt.show()
sns.violinplot(x="survival_status", y="positive_axillary_nodes", inner="quart", data=raw_data,size=5)
plt.title("Violen plot depicting the num_pos_axillary_nodes distribution")
plt.show()
sns.jointplot(x="Age", y="positive_axillary_nodes", data=raw_data, kind="kde");
plt.show();
sns.jointplot(x="Age", y="year", data=raw_data, kind="kde");
plt.show();
sns.jointplot(x="year", y="positive_axillary_nodes", data=raw_data, kind="kde");
plt.show();
