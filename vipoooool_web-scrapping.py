import os
os.listdir("../input/")
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
import pandas as pd
Url = "https://www.flipkart.com/search?q=iphone&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&as-pos=0&as-type=HISTORY"
url_client = urlopen(Url)
html = url_client.read()
url_client.close()
soup_obj = soup(html,"html.parser")
containers = soup_obj.findAll("div",{"class":"_1UoZlX"})
print(len(containers))
# Return overall html under that containers
print(soup.prettify(containers[0]))
container = containers[0]
Title = container.findAll("div",{"class":"_3wU53n"})
print(Title[0].text)
Price = container.findAll("div",{"class":"_1vC4OE _2rQ-NK"})
print(Price[0].text)
Rating = container.findAll("div",{"class":"hGSR34 _2beYZw"})
print(Rating[0].text)
filename = "Iphone.csv"
f = open(filename,'w')
headers = "Product_Name,Pricing,Rating\n"
f.write(headers)
for container in containers:
    product_name = container.findAll("div",{"class":"_3wU53n"})
    product = product_name[0].text
    
    price_container = container.findAll("div",{"class":"_1vC4OE _2rQ-NK"})
    price = price_container[0].text.strip()
    
    rating_container = container.findAll("div",{"class":"hGSR34 _2beYZw"})
    rating = rating_container[0].text
    
    # String Parsing
    
    trim_price = ''.join(price.split(',')) 
    # This line split when "," encounter and join with  spilt each part
    # ₹17,999 is split into ₹17 and 999 
    # Due to ''.join function it join that split part so output is ₹17999
    rm_rupee = trim_price.split("₹")
    final_price = "Rs. " + rm_rupee[1]

    
    split_rating = rating.split(" ")
    final_rating = split_rating[0]
    
    
    # Comma is replaced by ∣ because when comma encounter then CSV create another column
    print(product.replace(","," ∣") + " , " + final_price + " , " + final_rating+"\n")
    f.write(product.replace(","," ∣") + " , " + final_price + " , " + final_rating+"\n")
    
f.close()    
#reading recently created csv file
details = pd.read_csv("../input/Iphone.csv")
details.head()