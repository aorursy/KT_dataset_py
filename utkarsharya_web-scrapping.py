# importing various libraries which will be used as follows.
import requests
from bs4 import BeautifulSoup
import pandas as pd
import schedule
import time
# making request to the given url and parsing it through beautiful soup library.
res = requests.get('https://happycredit.in/kotak-mahindra-bank-offers/?page=3')
soup = BeautifulSoup(res.text,'lxml')
soup.find_all('a',class_='coupon-cart others')
#defining a function to scrap the data from the concerned url and storing it in a pandas dataframe.

def job():
    
    offers = []
    company = []
    validity = []
    bank = []
    deal = []
    card = []
    
    data = soup.find_all('a',class_="coupon-cart others")
    for i in data:
        offers.append(i.h3.text)
        company.append(i.find_all('b')[0].text)
        validity.append(i.find_all('b')[1].text)
        bank.append(i.find_all('b')[2].text)
        deal.append(i.find_all('b')[3].text)
        card.append(i.find_all('b')[4].text)
    df = pd.DataFrame({
        'Offers':offers,
        'Company':company,
        'Validity': validity,
        'Bank':bank,
        'Deal Type':deal,
        'Card Type':card})
df
# scheduling the job using schedule library in place of crontab
schedule.every().hour.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
# saving the scrapped data as a csv file.
df.to_csv('offers.csv')