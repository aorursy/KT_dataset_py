import requests
from bs4 import BeautifulSoup
url = 'https://www.ratemyprofessors.com/ShowRatings.jsp?tid=941931'
page = requests.get(url)
page
soup = BeautifulSoup(page.text, "html.parser")
proftags = soup.findAll("span", {"class": "Tag-bs9vf4-0" })
proftags
for mytag in proftags:
    print(mytag.get_text())
