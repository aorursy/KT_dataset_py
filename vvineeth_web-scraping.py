# Importing libraries that are used in web scraping

from bs4 import BeautifulSoup as soup

from urllib.request import urlopen as uReq

from lxml.html import parse



# Picking flipkark website link with iphone product

my_url = "https://www.flipkart.com/search?q=iphone&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off"



# Reading url and parsing the web page

uClient = uReq(my_url)

page_html = uClient.read()

uClient.close()

page_soup = soup(page_html, "html.parser")
# go and search the web page where all products contains in one page

containers = page_soup.findAll('div', {'class' : 'bhgxx2 col-12-12'})

print(len(containers))
# Printing Price and Rating in the form of html page

print(soup.prettify(containers[2]))
# Storing the above data with different variable

container = containers[2]
# Let's see the first iphone in that container html page

print(container.div.img["alt"])
# The price of Apple iPhone SE (Black, 64 GB)

price = container.findAll('div', {'class' : '_1uv9Cb'})

print(price[0].text)
# Rating of Apple iPhone SE (Black, 64 GB) 

rating  = container.findAll('div', {'class' : 'niH0FQ _36Fcw_'})

print(rating[0].text)