from urllib.request import urlopen as req
from bs4 import BeautifulSoup as soup
import numpy as np
import pandas as pd
import re
df={'Description':[],'Pro?':[],'Unlock?':[], 'Max?':[], 'Color':[], 'GB':[], 'Price':[]} # create a dictionary for a future dataframe
colors=['Space Gray', 'Midnight Green', 'Gray', 'Gold', 'Silver', 'Green', 'Silver', 'Purple', 'Black', 'Yellow', 'Red', 'White']
GB=['256GB', '512GB', '64GB', '128GB']
# here I used bs4,urlopen for parsing laptops description from the web-site removing with replace method all unnecessary symbols in the df
for n in range (1,14):
    my_url=('https://www.backmarket.com/search?page='+str(n)+'&q=iphone%2011')
    req(my_url)
    uClient=req(my_url)
    page_html=uClient.read()
    uClient.close() 
    page_soup=soup(page_html, 'html.parser')
    # find corresponding classes and assign it to variables. containers contain prices, names - titles of ads and description - all description lines 
    containers=page_soup.findAll("div",{"class":"price secondary small"})
    description=page_soup.findAll("h2",{"class":'cBkpLHWDpS5Fk2ZVL3a2F undefined corXsD6fPFG0SkyfwImBZ _2qvE_7NYIQCfs6GTiLSlx- _2A6YwALDdPLD83U2DRjbHW'})
    # append description columns replacing all not necessary symbols
    for piece in description:
        if re.findall('iPhone 11',piece.text)==['iPhone 11']:
            words=[]
            for word in colors: 
                if word in piece.text:
                    words.append(word)  
              
            if words!=[]: df['Color'].append(max(words))
            else:
                df['Color'].append(0)
        
            if re.findall('Pro',piece.text)==['Pro']: df['Pro?'].append(1) 
            else: 
                df['Pro?'].append(0)
            if re.findall('Max',piece.text)==['Max']: df['Max?'].append(1) 
            else: 
                df['Max?'].append(0)
            if re.findall('Unlocked',piece.text)==['Unlocked']:df['Unlock?'].append (1) 
            else: 
                df['Unlock?'].append (0)
            hdd=[]
            for mb in GB: 
                if mb in piece.text:
                    hdd.append(mb)  
            if hdd!=[]: df['GB'].append(max(hdd))
            else:
                df['GB'].append(0)
    # append titles and prices
    for n,container in enumerate(containers):
        if re.findall('iPhone 11',description[n].text)==['iPhone 11']:
            df['Description'].append(description[n].text.replace("\n",""))
            df['Price'].append(int(container.text.replace("\n","").replace('$','').replace('\xa0','').replace(',','').replace('.',''))/100)
a=[]
for col in df.values():
    a.append(len(col))
print(a)
df=pd.DataFrame(df) # transform this list to a dataframe
print('shape of the dataframe (rows, cols) is',df.shape) # check the shape of df
for index, row in enumerate(df.iterrows()):
    if re.findall(r'Case', str(row))==['Case']:
        df.drop(index, inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv('filename.csv', index=False)