import string
import requests
from lxml import html
import re
import time
tmp1 = list(string.ascii_lowercase)
tmp2 = list(string.ascii_lowercase)
tmp2.append('0-9')

ending_list = []

for c1 in tmp1:
      for c2 in tmp2:
        ending_list.append(c1+c2)

print(ending_list)
drugs = []

for page_num in ending_list:
    url = 'https://www.drugs.com/alpha/' + page_num + '.html'
    r = requests.get(url, allow_redirects=False)
    if r.status_code == 200:
        edited_html = (r.content).decode('utf-8')
        tree = html.fromstring(edited_html)
        p = tree.xpath('//*[@id="content"]/div[2]/ul/li/a/text()')
        for drug in p:
            drugs.append(drug)

print(len(drugs))
MyFile=open('drugs.txt','w')

for element in drugs:
    MyFile.write(element)
    MyFile.write('\n')
MyFile.close()