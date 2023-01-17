import numpy as np 
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup


def getHTMLContent(link):
    html = urlopen(link)
    soup = BeautifulSoup(html, 'html.parser')
    return soup
content = getHTMLContent('https://money.usnews.com/careers/best-jobs/rankings/the-100-best-jobs')
articles = content.find_all('div')
for article in articles:
    print(article.text)
article = content.find('div', class_= 'DetailCardJob__Layout-jjqty-0 hZKOTD')
rows = content.find_all('tr')
article