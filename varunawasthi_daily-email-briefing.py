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
import requests
import bs4
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os.path
print("Today's COVID-19 cases")

b = requests.get("https://www.worldometers.info/coronavirus/")
soup = bs4.BeautifulSoup(b.text,"html.parser")
cases_w = soup.find_all("div", attrs={"class":"maincounter-number"})[0].text
print("Worldwide confirmed cases:",cases_w)

b = requests.get("https://www.worldometers.info/coronavirus/country/us/")
soup = bs4.BeautifulSoup(b.text,"html.parser")
cases_US = soup.find_all("div", attrs={"class":"maincounter-number"})[0].text
print("US confirmed cases:",cases_US)

b = requests.get("https://www.worldometers.info/coronavirus/country/us/")
soup = bs4.BeautifulSoup(b.text,"html.parser")
cases_d = soup.find_all("div", attrs={"class":"maincounter-number"})[1].text
print("US deaths:",cases_d)


#x = cases_US.replace(",","") #need remove comma before converting it to integer
#print(int(x))


b = requests.get("https://www.worldometers.info/coronavirus/")
soup = bs4.BeautifulSoup(b.text,"html.parser")
last = soup.find_all("div", attrs={"style":"font-size:13px; color:#999; margin-top:5px; text-align:center"})[0].text
print("This was",last,"\n")
print("Today's Market Summary:")
#print("Today's Market Summary:")
#this function removes all the text after the percentage change
def fix(x):
    sep1 = "At"
    sep2 = "As"
    temp= x.split(sep1,1)[0]
    temp =temp.split(sep2,1)[0]
    temp = temp.replace("-"," -")
    temp = temp.replace("+"," +")
    temp = temp.replace("(","")
    return temp.replace(")","")


s =requests.get("https://finance.yahoo.com/chart/%5EGSPC")
soup = bs4.BeautifulSoup(s.text,"html.parser")
prices_SP = soup.find_all("div", attrs={"class":"D(ib) Fw(200) Mend(25px) Mend(20px)--lgv3"})[0].text
prices_SP = prices_SP.replace("-"," -")
print("SP 500:", fix(prices_SP))



s =requests.get("https://finance.yahoo.com/chart/%5EDJI")
from bs4 import BeautifulSoup
soup = bs4.BeautifulSoup(s.text,"html.parser")
prices_D = soup.find_all("div", attrs={"class":"D(ib) Fw(200) Mend(25px) Mend(20px)--lgv3"})[0].text
print("DJIA:", fix(prices_D))


s =requests.get("https://finance.yahoo.com/chart/%5EIXIC")
from bs4 import BeautifulSoup
soup = bs4.BeautifulSoup(s.text,"html.parser")
prices_N = soup.find_all("div", attrs={"class":"D(ib) Fw(200) Mend(25px) Mend(20px)--lgv3"})[0].text
print("NASDAQ:", fix(prices_N))

s =requests.get("https://finance.yahoo.com/quote/CL%3DF?p=CL%3DF")
from bs4 import BeautifulSoup
soup = bs4.BeautifulSoup(s.text,"html.parser")
prices_C = soup.find_all("div", attrs={"class":"D(ib) Mend(20px)"})[0].text
print("Crude Oil: $", fix(prices_C),"\n")
def wfix(a):
    sep = "Feels"
    temp1 = a.split(sep, 1)[0]
    temp0 = temp1.replace("F", " F")
    temp = temp0.replace("°", "°F ")

    return temp

weather_options = ["Champaign IL","Weston CT", "New York, NY","Chicago IL"]
weather_links = ["https://bit.ly/39ivfPd","https://bit.ly/2TMGc6j","https://bit.ly/2IG7TYa","https://bit.ly/2xsPpIf"]



print("Today's Weather in",weather_options[1],"is :",'\n')
#t = requests.get("https://bit.ly/39ivfPd") #this is the weather for Champaign IL from the weather channel website
t = requests.get(weather_links[1]) #this is the weather for Weston CT from the weather Channel website
from bs4 import BeautifulSoup
soup = bs4.BeautifulSoup(t.text,"html.parser")

weather = soup.find_all("div",attrs={"data-testid":"wxPhrase"})[0].text
temperature = soup.find_all("span",attrs={"data-testid":"TemperatureValue"})[0].text
temp_feel = soup.find_all("div",attrs={"data-testid":"FeelsLikeSection"})[0].text
temp_high = soup.find_all("span",attrs={"data-testid":"TemperatureValue"})[1].text
temp_low = soup.find_all("span",attrs={"data-testid":"TemperatureValue"})[2].text
precipitation = soup.find_all("div",attrs={"data-testid":"SegmentPrecipPercentage"})[1].text

print("Weather:",weather)
print("Temperature:",wfix(temperature),"and Feels like",wfix(temp_feel))
print("High/Low:",wfix(temp_high),"/",wfix(temp_low))
print("Precipitation chance:",precipitation)

from datetime import date
import time
from datetime import datetime
import calendar
my_date = date.today()
date = time.strftime("%m/%d/%Y")
day_of_week = calendar.day_name[my_date.weekday()]


email = 'enter your email address'
password = 'enter your email password'
send_to_email = 'add the email you want to send this to'
subject = 'Today\'s'     + str(date) +"      " + str(day_of_week)

str1 = "Covid-19 News: " +"\n" + "Worldwide confirmed cases:" + cases_w +"\n" + "U.S confirmed cases:" + cases_US +"\n"
str2 = "U.S confirmed deaths:" + cases_d +"\n" + "This was: " + last + "\n" + "\n"

str3 = "Market Summary:" + "\n" + "SP 500:  " + fix(prices_SP) + "\n" + "DJIA:  " + fix(prices_D) + "\n"
str4 = "NASDAQ:  " + fix(prices_N) + "\n" + "Crude Oil $:  " + fix(prices_C) + "\n" + "\n"

str5 = "Weather in Weston CT: " +"\n"
str6 = "Weather: " + weather +"\n"
str7 = "Temperature: "+ wfix(temperature) + "and Feels like "+ wfix(temp_feel)+"\n"
str8 = "High/Low: "+ wfix(temp_high) + "/ "+ wfix(temp_low)+"\n"
str9 = "Precipitation chance: "+ precipitation + "\n"



message = str1 + str2 + str3 +str4+ str5 +str6 + str7 + str8 + str9

msg = MIMEMultipart()
msg['From'] = email
msg['To'] = send_to_email
msg["Subject"] = subject

msg.attach(MIMEText(message, "plain"))
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(email, password)
text = msg.as_string()
server.sendmail(email, send_to_email, text)
server.quit()