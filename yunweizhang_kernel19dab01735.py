# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#read files and print head
shop=pd.read_csv('/kaggle/input/ptr-rd1/Extra_material_2.csv')
order=pd.read_csv('/kaggle/input/ptr-rd1/Extra_material_3.csv')

#Remove duplicates from the two files
shop=shop.drop_duplicates()
order=order.drop_duplicates()


# Create a brand series and sort it in ascending order

shop['brand_lower']=shop.brand.str.lower()
brand=shop[['brand','brand_lower']].drop_duplicates().sort_values(by='brand_lower',ascending=True)
brand.drop(columns='brand_lower',inplace=True)
brand=brand['brand']

#check unique values of shop type
#print(shop.shop_type.unique())


#Replace 'Official Shop (not mall)' with 'Official Shop'
shop.shop_type=shop['shop_type'].replace('Official Shop (not mall)','Official Shop')
print(shop.shop_type.unique())

#Merge the two files 
merged=pd.merge(order, shop, left_on='shopid',right_on='shop_id')

#Filter by shop type='Official Shop'
official=lambda x: x=='Official Shop'
official_order=merged[merged.shop_type.map(official)]
# Filter by date - 10th May to 31st May 2019
official_order['date_id']=pd.to_datetime(official_order['date_id'], format="%d/%m/%Y")
start_date = pd.to_datetime('10/05/2019', format="%d/%m/%Y")
end_date = pd.to_datetime('31/05/2019', format="%d/%m/%Y")

date_check=lambda x: (x>=start_date) & (x <=end_date)
official_order=official_order[official_order.date_id.map(date_check)]
official_order
#calculate sales for each item
official_order['sales']=official_order['amount']*official_order['item_price_usd']
official_order[:1]
#official_order = official_order.groupby(["brand", "itemid"])["amount"].sum().reset_index()

#official_order["RN"] = (official_order.sort_values("amount", ascending=False).groupby("brand").cumcount() + 1).astype(int)
#official_order=official_order[official_order['RN']<=3]
#official_order
# Return top 3 items for a brand

def generate_top_3(brand_name):
    brand_sales={}
    for ind in official_order.index:
        if official_order.loc[ind,'brand']==brand_name:
            if official_order.loc[ind,'itemid'] in brand_sales:
                brand_sales[official_order.loc[ind,'itemid']]+=official_order.loc[ind,'sales']
            else:
                brand_sales[official_order.loc[ind,'itemid']]=official_order.loc[ind,'sales']
    brand_sales=pd.DataFrame.from_dict(brand_sales, orient='index',columns=['sales'])
    brand_sales=brand_sales.sort_values(by='sales', ascending=False)
    return brand_sales[:3]



result=pd.DataFrame(columns=['Index','Answers'])

ind=1

for index, brand_name in brand.items():
    brand_sales=generate_top_3(brand_name)
    
    Answer=[brand_name]
    
    
    if (brand_sales.empty):
        Answer.append('N.A')
      
    else:
        for i in brand_sales.index:
            Answer.append(str(i))
    Answer=', '.join(Answer)
    #print(type(Answer))

    #result=result.append(pd.DataFrame(data={'Index':ind,'Answers': Answer}),ignore_index=True)
    result=result.append(pd.DataFrame(data={'Index':ind,'Answers': [Answer]}),ignore_index=True)
    ind=ind+1
    
print(result)    

    
result.to_csv("output.csv",index=False)  