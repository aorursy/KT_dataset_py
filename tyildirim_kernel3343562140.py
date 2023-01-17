import numpy as np
import pandas as pd
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
url= 'https://www.levels.fyi/comp.html?track=Data%20Scientist#'

page=requests.get(url)


soup=BeautifulSoup(page.content)

company=[]
company_name=soup.find_all('span',{''})
company_name
