import pandas as pd
import numpy as np
import seaborn as sns
application = pd.read_csv("../input/application_train.csv")
# application.head()
# application.columns.values
bureau = pd.read_csv("../input/bureau.csv")
# bureau.head()

merged = pd.merge(application,bureau,on = 'SK_ID_CURR')
###Type of loans taken by different income types

a = application.groupby(by = [ 'NAME_CONTRACT_TYPE','NAME_INCOME_TYPE']).size()
print(a)

####DEFAULTERS ACCORDING TO INCOME TYPE
b = application.groupby(by = [ 'TARGET','NAME_INCOME_TYPE']).size()
print(b)
# for each in b:
#     print(each)
b.index

c = application['NAME_INCOME_TYPE']
c.value_counts()


d = application['CODE_GENDER']
d.value_counts()

e = application['TARGET']
e.value_counts()


####DEFAULTERS GENDER WISE
f = application.groupby(by = [ 'TARGET','CODE_GENDER']).size()
print(f)
