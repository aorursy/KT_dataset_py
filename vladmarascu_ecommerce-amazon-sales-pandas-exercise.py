import pandas as pd

ecom=pd.read_csv('../input/ecommerce-purchases-csv/Ecommerce Purchases.csv')
ecom.head(3)
#len(ecom.columns)

#len(ecom.index)

ecom.info()
ecom['Purchase Price'].mean()
ecom['Purchase Price'].max()
ecom['Purchase Price'].min()
len(ecom[ecom['Language']=='en'])

# ecom[ecom['Language']=='en']['Language'].count()
len(ecom[ecom['Job']=='Lawyer'].index)
ecom['AM or PM'].value_counts()
ecom['Job'].value_counts().head(5)
ecom[ecom['Lot']=='90 WT']['Purchase Price']
ecom[ecom['Credit Card']==4926535242672853]['Email']
len(ecom[(ecom['CC Provider']=='American Express') & (ecom['Purchase Price']>95)])
sum(ecom['CC Exp Date'].apply(lambda exp: exp[3:]=='25'))
ecom['Email'].apply(lambda email: email.split('@')[1]).value_counts().head(7)