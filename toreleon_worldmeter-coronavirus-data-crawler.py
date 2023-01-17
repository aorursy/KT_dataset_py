!pip install js2xml

import numpy as np

import pandas as pd

import requests

from bs4 import BeautifulSoup as bs

import js2xml
def crawler(url):

    soup = bs(requests.get(url).content, 'html.parser')

    dataframe = pd.DataFrame()

    for script in soup.find_all('script'):

        if 'Highcharts' in script.text:

            parser = js2xml.parse(script.text)

            date = parser.xpath('//property[@name="categories"]//string/text()')

            title = parser.xpath('//property[@name="title"]//string/text()')

            data = parser.xpath('//property[@name="data"]//number/@value')

            if 'Daily Deaths' in title:

                dataframe['Date'] = date

                dataframe['Daily Deaths'] = data

            if 'New Cases vs. New Recoveries' in title:

                newcases = data[len(date)::]

                dataframe['New Recoveries'] = data[0:len(date)]

                while(len(newcases)<len(date)): #Checking for missing data

                    newcases.insert(0,0)

                    dataframe['Daily New Cases'] = newcases

    return dataframe
#url format: https://www.worldometers.info/coronavirus/country/country_name

crawler('https://www.worldometers.info/coronavirus/country/philippines/')