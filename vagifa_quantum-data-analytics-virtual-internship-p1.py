import pandas as pd
import numpy as np
import plotly.express as px
from string import punctuation
transaction_data = pd.read_excel('https://insidesherpa.s3.amazonaws.com/vinternships/companyassets/32A6DqtsbF7LbKdcq/QVI_transaction_data.xlsx')
customer_data = pd.read_csv('https://insidesherpa.s3.amazonaws.com/vinternships/companyassets/32A6DqtsbF7LbKdcq/QVI_purchase_behaviour.csv')
transaction_data.head(5)
transaction_data.info()
transaction_data.describe()
transaction_data['DATE'] = pd.to_datetime(transaction_data['DATE'],errors='coerce',unit='d',origin='1900-01-01')
transaction_data['PROD_NAME'].value_counts()
chips = transaction_data[transaction_data['PROD_NAME'].str.contains('Salsa') == False].copy()
chips.head(5)
def remove_punc_digit_no_g(val):
    val = [v for v in val if v not in punctuation]
    result = ''.join([i for i in val if not i.isdigit()])
    final = ''.join([e for e in result if e.isalnum() or e == ' '])
    return final.replace('g','')
chips['PROD_NAME'].apply(lambda x:remove_punc_digit_no_g(x)).str.split(expand=True).stack().value_counts()
def remove_punc_digit(val):
    val = [v for v in val if v not in punctuation]
    result = ''.join([i for i in val if not i.isdigit()])
    final = ''.join([e for e in result if e.isalnum() or e == ' '])
    return final
chips['PROD_NAME'] = chips['PROD_NAME'].apply(lambda x:remove_punc_digit(x))
chips.describe()
chips[chips['PROD_QTY'] == 200]
chips[chips['LYLTY_CARD_NBR'] == 226000]
final = chips[chips['LYLTY_CARD_NBR'] != 226000]
customer_data.info()
#merged = pd.merge(customer_data,transaction_data,on='LYLTY_CARD_NBR')
bydate = final.groupby('DATE')[['TXN_ID']].count().reset_index()
bydate
fig = px.line(bydate,bydate['DATE'],bydate['TXN_ID'])
fig.update_layout(title='Transactions over time',title_x=0.5)
december = bydate[bydate['DATE'].isin(list(bydate['DATE'][151:181]))]
fig = px.bar(december,december['DATE'],december['TXN_ID'])
fig.update_layout(title='Transactions of December 2018',title_x=0.5)
def pack_size(inp):
    inp = ''.join([x for x in inp if x.isdigit()]) 
    return str(inp + 'g')
transaction_data['Pack Size'] = transaction_data['PROD_NAME'].apply(lambda d:pack_size(d))
transaction_data['Pack Size'].value_counts()
transaction_data.head(5)
fig = px.histogram(transaction_data['Pack Size'])
fig.update_layout(title='Histogram of Transaction Data',title_x=0.5)
transaction_data['Brand Name'] = transaction_data['PROD_NAME'].str.split().apply(lambda x:x[0])
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('Red','RRD')
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('SNBTS','SUNBITES')
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('infzns','Infuzions')
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('WW','woolworths')
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('Smith','Smiths')
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('NCC','Natural')
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('Dorito','Doritos')
transaction_data['Brand Name'] = transaction_data['Brand Name'].replace('Grain','GrnWves')
transaction_data['Brand Name'].value_counts()


#bypack = cleaned.groupby('Pack Size')[['TXN_ID']].count().reset_index()



customer_data.info()
customer_data.describe()
customer_data['LIFESTAGE'].value_counts()
customer_data['PREMIUM_CUSTOMER'].value_counts()
merged = pd.merge(transaction_data,customer_data,on='LYLTY_CARD_NBR')
merged.isnull().sum()
byc = merged.groupby(['LIFESTAGE','PREMIUM_CUSTOMER'])[['TOT_SALES']].sum().reset_index()
fig = px.bar(byc,byc['LIFESTAGE'],byc['TOT_SALES'],byc['PREMIUM_CUSTOMER'],text=(byc['TOT_SALES']))
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(title='Proportion of Sales',title_x=0.5)

fig.show()
byc = merged.groupby(['LIFESTAGE','PREMIUM_CUSTOMER'])[['LYLTY_CARD_NBR']].count().reset_index()
fig = px.bar(byc,byc['LIFESTAGE'],byc['LYLTY_CARD_NBR'].unique(),byc['PREMIUM_CUSTOMER'],text=byc['LYLTY_CARD_NBR'].unique())
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(title='Proportion of Customers',title_x=0.5)
fig.show()
unique_cust = merged[merged['LYLTY_CARD_NBR'].isin(list(pd.unique(merged['LYLTY_CARD_NBR'])))].copy()
byc = unique_cust.groupby(['LIFESTAGE','PREMIUM_CUSTOMER'])[['PROD_QTY','LYLTY_CARD_NBR']].sum().reset_index()
byc['AVG'] = byc['PROD_QTY'] / byc['LYLTY_CARD_NBR']
fig = px.bar(byc,byc['LIFESTAGE'],byc['AVG'],byc['PREMIUM_CUSTOMER'])
# Change the bar mode
fig.update_layout(barmode='group',title='AVG Units per customer',title_x=0.5)
fig.show()
byc = merged.groupby(['LIFESTAGE','PREMIUM_CUSTOMER'])[['TOT_SALES','PROD_QTY']].sum().reset_index()
byc['AVG'] = byc['TOT_SALES'] / byc['PROD_QTY']
fig = px.bar(byc,byc['LIFESTAGE'],byc['AVG'],byc['PREMIUM_CUSTOMER'])
fig.update_layout(barmode='group',title='AVG Sales per Unit',title_x=0.5)
fig.show()
import scipy.stats
merged['PricePerUnit'] = merged['TOT_SALES'] / merged['PROD_QTY']
filter1 = merged[(merged['LIFESTAGE'].isin(["YOUNG SINGLES/COUPLES", "MIDAGE SINGLES/COUPLES"]))  & (merged['PREMIUM_CUSTOMER'] == 'Mainstream')]
filter2 = merged[(merged['LIFESTAGE'].isin(["YOUNG SINGLES/COUPLES", "MIDAGE SINGLES/COUPLES"]))  & (merged['PREMIUM_CUSTOMER'] != 'Mainstream')]
filter2
scipy.stats.ttest_ind(filter1['PricePerUnit'],filter2['PricePerUnit'])
