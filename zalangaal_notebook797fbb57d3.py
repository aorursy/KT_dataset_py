import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
oldal_linkek=[]
oldal_linkek.append('https://auto.jofogas.hu/magyarorszag/auto?re=1999&rs=1999&o=2')
for x in range(3,22):
    url='https://auto.jofogas.hu/magyarorszag/auto?re=1999&rs=1999&o=' + str(x) 
    oldal_linkek.append(url)


c1=[]
c2=[]
c3=[]
c4=[]
c5=[]

for n in range(len(oldal_linkek)):
    
    url=oldal_linkek[n]
    response = requests.get(url)
    soup=BeautifulSoup(response.text, 'html.parser')
    
    results1=soup.find_all('span',attrs={'class','price-value'})

    results2=soup.find_all('span',attrs={'class','vehicle-brand'})

    results3=soup.find_all('span',attrs={'class','vehicle-model'})
    
    results4=soup.find_all('div',attrs={'class','vehicle-fuel'})
    
    

    for i in range(len(results1)):
        text=results1[i].get_text().strip()
        c1.append(text)
    
    for i in range(len(results2)):
        text=results2[i].get_text().strip()
        c2.append(text)

    for i in range(len(results3)):
        text=results3[i].get_text().strip()
        c3.append(text)

    for i in range(len(results4)):
        text=results4[i].get_text().strip()
        c4.append(text)
tabla_kocsik={"Ár forintban":c1,
            "Márka":c2,
            "Típus":c3,
            "Üzemanyag":c4}
df_kocsik = pd.DataFrame.from_dict(tabla_kocsik)
print(df_kocsik)
writer = pd.ExcelWriter('kocsik.xlsx', engine='xlsxwriter')
df_kocsik.to_excel(writer, sheet_name='Kocsik 1999', index=False)
writer.save()