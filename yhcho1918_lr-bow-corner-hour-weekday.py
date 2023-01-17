import pandas as pd
import numpy as np
df_train = pd.read_csv('../input/X_train.csv', encoding='cp949')
df_test = pd.read_csv('../input/X_test.csv', encoding='cp949')
y_train = pd.read_csv('../input/y_train.csv').gender
IDtest = df_test.custid.unique()

df_train.head()
df_all = pd.concat([df_train, df_test])
df_all['sales_hour'] = df_all['sales_time']//100;
df_all['sales_wkday'] = pd.to_datetime(df_all.sales_date).dt.weekday

def makeBOW(col):
    f = lambda x: np.where(len(x) >=1, 1, 0)

    train = pd.pivot_table(df_all, index='custid', columns=col, values='tot_amt',
                             aggfunc=f, fill_value=0).reset_index(). \
                             query('custid not in @IDtest').drop(columns=['custid'])
    test = pd.pivot_table(df_all, index='custid', columns=col, values='tot_amt',
                             aggfunc=f, fill_value=0).reset_index(). \
                             query('custid in @IDtest').drop(columns=['custid'])
    return train, test
train1, test1 = makeBOW('corner_nm')
train2, test2 = makeBOW('sales_hour')
train3, test3 = makeBOW('sales_wkday')

X_train = pd.concat([train1, train2, train3], axis=1).values
X_test = pd.concat([test1, test2, test3], axis=1).values
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict_proba(X_test)[:,1]
fname = 'submissions.csv'
submissions = pd.concat([pd.Series(IDtest, name="custid"), pd.Series(pred, name="gender")] ,axis=1)
submissions.to_csv(fname, index=False)
print("'{}' is ready to submit." .format(fname))