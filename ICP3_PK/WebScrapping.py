# Importing requests library to send HTTP/1.1 requests
import requests

# Variable assigned to url of website from which data is to be scrapped
wikipage = requests.get('https://en.wikipedia.org/wiki/Deep_learning').text

# Parsing daata using BeautifulSoup function
from bs4 import BeautifulSoup
soup = BeautifulSoup(wikipage, 'html')
print(soup.prettify())

# Displaying the title of the web page link
print(soup.title.string)

# Finding all the links within the page containing a tag
alinks = soup.find_all("a")
for link in alinks:
    print(link.get('href'))