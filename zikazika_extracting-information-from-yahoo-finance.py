!pip install selenium
from selenium import webdriver

from selenium.webdriver.common.by import By

from selenium.webdriver.common.keys import Keys
import pandas as pd
import time

import datetime
import smtplib

from email.mime.multipart import MIMEMultipart
import sys

sys.path.append(r"C:\Users\Noah\Desktop\zzz\chromedriver.exe")
driver = webdriver.Chrome(executable_path=r"C:\Users\Noah\Desktop\zzz\chromedriver.exe")
from IPython.display import Image

Image("../input/1.png")
stocks_gainers='//*[@id="SecondaryNav-0-SecondaryNav-Proxy"]/div/ul/li[5]/a'

stocks_losers='//*[@id="SecondaryNav-0-SecondaryNav-Proxy"]/div/ul/li[6]/a'
# clicking on the losers,gainers

def market_chooser(market_product):

    

    try:

        market_product_type = driver.find_element_by_xpath(market_product)

        market_product_type.click()

    except Exception as e:

        pass
df = pd.DataFrame()

def compile_data():

    global df

    global top_of_the_list

    

    top_of_the_list = driver.find_elements_by_xpath('//*[@id="scr-res-table"]/div[1]/table/tbody/tr[1]/td[2]')

    top_of_the_list_list = [value.text for value in top_of_the_list]



   



    for i in range(len(top_of_the_list_list)):

        try:

            df.loc["Characteristics", i] = top_of_the_list_list[i]

        except Exception as e:

            pass

    print('Excel Sheet Created!')
username = 'noahweber@gmail.com'

password = 'xxxxxxxxxxxxxxxxxxx'
def connect_mail(username, password):

    global server

    server = smtplib.SMTP('smtp.gmail.com', 587)

    server.ehlo()

    server.starttls()

    server.login(username, password)
def create_msg():

    global msg

    msg = '\nInformation: {}\n'.format(current_values)
def send_email(msg):

    global message

    message = MIMEMultipart()

    message['Subject'] = 'Yahoo Finance'

    message['From'] = 'noahweber@gmail.com'

    message['to'] = 'noahweber@gmail.com'



    server.sendmail('noahweber@gmail.com', 'noahweber@gmail.com', msg)
for i in range(8):    

    link = 'https://finance.yahoo.com/'

    driver.get(link)

    #wait for the page to load

    time.sleep(10)



    markets_button_yfinance = driver.find_element_by_xpath('//*[@id="Nav-0-DesktopNav"]/div/div[3]/div/div[1]/ul/li[4]/a')

    markets_button_yfinance.click()

    market_chooser(stocks_gainers)

    

    

    compile_data()

    

    #save values for email

    current_values = df.iloc[0]

    

    print('Number of iterations: {}'.format(i))

    create_msg()

    connect_mail(username,password)

    send_email(msg)

    print('Email sent')

    

    df.to_excel('best_worst_stock.xlsx')

    

    time.sleep(60)