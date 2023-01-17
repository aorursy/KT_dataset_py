!pip install requests
!pip install beautifulsoup4
import re
import csv
import requests
from bs4 import BeautifulSoup

resp=requests.get("https://www.imdb.com/list/ls068010962/")

soup = BeautifulSoup(resp.text, 'html.parser')
results = soup.find_all('div', attrs={'class':'lister-item-content'})
len(results)
results[0:3]
first_result = results[0]
first_result
first_result.find('a').text[0:-1]
result2=soup.find('img', {'src':re.compile('.jpg')})['src']
first_result.find('p').text[1:-1].strip()[0:7].replace("|"," ")
first_result.find('p').text[1:-1].strip()[9:]
records = []
for result in results:
    Name = result.find('a').text[0:-1]
    Image_url = soup.find('img', {'src':re.compile('.jpg')})['src']
    celebrity =result.find('p').text[1:-1].strip()[0:7].replace("|"," ")
    movie = result.find('p').text[1:-1].strip()[9:]
    records.append((Name, Image_url, celebrity, movie))
len(records)
records[0:3]
import pandas as pd
df = pd.DataFrame(records, columns=[Name, Image_url, celebrity, movie])
df.loc[0]
df=df.rename(columns={" Navin Nischol":"Name","https://m.media-amazon.com/images/M/MV5BMjAwMjk3NDUzN15BMl5BanBnXkFtZTcwNjI4MTY0NA@@._V1_UX140_CR0,0,140,209_AL_.jpg":"Image_url","Zorro":"Movie Name"})
df.head()
df.tail()