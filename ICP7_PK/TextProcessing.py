# Importing requests library to send HTTP/1.1 requests
import requests
import numpy
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
from nltk import ngrams
nltk.download()

# Variable assigned to url of website from which data is to be scrapped
googlePage = requests.get('https://en.wikipedia.org/wiki/Google').text

# Parsing data using BeautifulSoup function
soup = BeautifulSoup(googlePage, 'html.parser')
#print(soup.prettify())

# Displaying the title of the web page link
print('Title of the web page: ', soup.title.string)

# Saving the data extracted from web link into the text file titled 'input'
file = open('input' + '.txt', 'a+', encoding='utf-8')
body = soup.find('div', {'class': 'mw-parser-output'})
file.write(str(body.text))

with open('input.txt', 'r', encoding='utf8') as inputData:
    content = inputData.read().replace('\n', '')

# Tokenization
token = nltk.word_tokenize(content)
print('Tokens identified are', token)

# Part Of Speech tagging (POS)
pos = nltk.pos_tag(token)
print('Part of Speech associated with the tokens identified are', pos)

# Stemming - identifying the root or base word of the terms associated
pStemmer = PorterStemmer()
for x in token:
    print('Result of Stemming using PorterStemmer for ', x, 'is ', pStemmer.stem(x))

lStemmer = LancasterStemmer()
for y in token:
    print('Result of Stemming using LancasterStemmer for ', y, 'is ', lStemmer.stem(y))

sStemmer = SnowballStemmer('english')
for z in token:
    print('Result of Stemming using SnowballStemmer for ', z, 'is', sStemmer.stem(z))

# Lemmatization - noramlization of text based on the meaning as part of the speech (converts plurals or adjective to their basic, meaningful singular form)
lemmatizer = WordNetLemmatizer()
print('Result of Lemmatization: ', lemmatizer.lemmatize(x))

# Trigram
trigram = ngrams(content.split(), 3)
for gram in trigram:
    print('Trigram data is ', gram)
print(str(trigram))

# Named Entity Recognition
print('Named Entitiy Reoognition is ', ne_chunk(pos_tag(wordpunct_tokenize(content))))