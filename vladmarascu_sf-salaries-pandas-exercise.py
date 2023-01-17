import pandas as pd
sal=pd.read_csv('../input/salaries-num/Salaries.csv')
sal.head()
sal.info()
sal['BasePay'].mean()
sal['OvertimePay'].max()
sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']
sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits']
sal[sal['TotalPayBenefits']==sal['TotalPayBenefits'].max()]['EmployeeName']
sal['TotalPayBenefits'].idxmax() # OR .argmax()

sal.loc[sal['TotalPayBenefits'].idxmax()] # OR iloc/argmax
sal.iloc[sal['TotalPayBenefits'].argmin()]
sal.groupby('Year').mean()['BasePay']
sal['JobTitle'].nunique()
sal['JobTitle'].value_counts().head(5)
sum(sal[sal['Year']==2013]['JobTitle'].value_counts()==1)
def chief_string(title):

    if 'chief' in title.lower().split():

        return True

    else:

        return False
sum(sal['JobTitle'].apply(lambda x: chief_string(x)))
sal['title_length']=sal['JobTitle'].apply(len) # create new column with title lengths
sal[['TotalPayBenefits','title_length']].corr()