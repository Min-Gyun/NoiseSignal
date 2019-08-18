#!/usr/bin/env python
# -*- coding: utf-8 -*-
# •The SEC limits users to 10 requests per second, so we need to make sure we're not making requests too quickly.

def get_uncertainty_on_filing (URL):
    import urllib2

    CIK = URL.split("/")[-2]
    try:
        response3 = urllib2.urlopen(URL, timeout=5)
    except urllib2.HTTPError, e:
        print 'HTTPError = ' + str(e.code)
        response3 = ""

    words = ['anticipate', 'believe', 'depend', 'fluctuate', 'indefinite','likelihood', 'possible', 'predict', 'risk', 'uncertain']
    count={}

    for elem in words:
        count[elem] = 0

    for line in response3:
        elements = line.split()
        for word in words:
            count[word] = count[word]+elements.count(word)

    #print (CIK)
    #print (count)
    #print URL
    #print (Year)
    return count

# FORM 10-12B, S-1, F-1
def get_read_form():
    # Given a Master Index URL of EDGAR find the path of raw text filing
    import urllib2
    import time

    Year = '2019' #GIVEN
    QUARTERS = ['QTR1','QTR2','QTR3','QTR4']
    #QUARTERS = ['QTR1']
    #FILE='10-12B' #GIVEN
    FILE = '10-K'  # GIVEN

    ##### Get the Master Index File for the given Year
    # CIK|Company Name|Form Type|Date Filed|Filename
    for Qtr in QUARTERS:
        try:
            url='https://www.sec.gov/Archives/edgar/full-index/%s/%s/master.idx' %(Year, Qtr)
            response = urllib2.urlopen(url,timeout=5)
        except:
            exit()

        string_match1 = 'edgar/data/'

        # URL이 여러개 존재
        for line in response:
            if FILE in line:
                UNPACK_LINE = line.strip().split('|')
                CIK = UNPACK_LINE[0]
                NAME = UNPACK_LINE[1]
                FORM_TYPE = UNPACK_LINE[2]
                DATE_FILED = UNPACK_LINE[3]
                FILE_NAME = UNPACK_LINE[4]

                for element in UNPACK_LINE:
                    if string_match1 in element:
                        url3 = 'https://www.sec.gov/Archives/' + element
                        print Qtr, UNPACK_LINE
                        print DATE_FILED, url3
                        #print element2[-2], url3
                        print get_uncertainty_on_filing (url3)
                        time.sleep(1)

def get_read_form2 (TICKER):
    # Given a Master Index URL of EDGAR find the path of raw text filing
    import urllib2
    import time
    import pandas as pd

    #CIK = get_CIK (TICKER)

    CIK = MapTickerToCik (TICKER)

    #Year = '2019'  # GIVEN
    QUARTERS = ['QTR1', 'QTR2', 'QTR3', 'QTR4']
    # QUARTERS = ['QTR1']
    # FILE='10-12B' #SPIN OFF: 10-28B, IPO: S-1 or F-1, Annual/Quarter Report: 10-K, 10-Q

    ##### Get the Master Index File for the given Year
    # CIK|Company Name|Form Type|Date Filed|Filename

    CIKs = []
    NAMEs = []
    FORM_TYPEs = []
    DATE_FILEDs = []
    URLs = []

    for Year in range (2015, 2020):
        for Qtr in QUARTERS:
            try:
                url = 'https://www.sec.gov/Archives/edgar/full-index/%s/%s/master.idx' % (Year, Qtr)
                response = urllib2.urlopen(url, timeout=5)
            except:
                break

            string_match1 = 'edgar/data/'

            # URL이 여러개 존재
            for line in response:
                if CIK in line and ('10-K' in line or '10-Q' in line):
                    UNPACK_LINE = line.strip().split('|')
                    for element in UNPACK_LINE:
                        if string_match1 in element:
                            url3 = 'https://www.sec.gov/Archives/' + element

                            CIKs.append(int(UNPACK_LINE[0]))
                            NAMEs.append(UNPACK_LINE[1])
                            FORM_TYPEs.append(UNPACK_LINE[2])
                            DATE_FILEDs.append(UNPACK_LINE[3])
                            URLs.append(url3)

                            print Year, Qtr, UNPACK_LINE
                            #print element2[-2], url3
                            #print get_uncertainty_on_filing(url3)
                            #time.sleep(6)

    dfData = pd.DataFrame(data={"CIK": CIKs, "NAME": NAMEs, "FORM_TYPE": FORM_TYPEs, "DATE_FILED": DATE_FILEDs, "URL": URLs})

    return dfData

#get_read_form()
def get_TICKER (CIK):
    ##### DATA SOURCE
    ##### http://rankandfiled.com/#/data/tickers
    import os
    FILE_PATH = os.path.join(os.getcwd(), "cik_ticker.tmp")

    f = open (FILE_PATH)

    TICKER = "N/A"
    for line in f:
        if CIK == line.split('|') [0]:
            TICKER = line.split('|') [1]
            break

    f.close()
    return TICKER

def get_CIK (TICKER):
    ##### DATA SOURCE
    ##### http://rankandfiled.com/#/data/tickers
    import os
    FILE_PATH = os.path.join(os.getcwd(), "cik_ticker.tmp")

    f = open (FILE_PATH)

    CIK = 0
    for line in f:
        if TICKER.lower() == line.split('|') [1].lower():
            CIK = line.split('|') [0]
            break

    f.close()
    return CIK

# Reference: https://www.quantopian.com/posts/scraping-10-ks-and-10-qs-for-alpha
def MapTickerToCik(ticker):
    import re
    import requests

    url = 'http://www.sec.gov/cgi-bin/browse-edgar?CIK={}&Find=Search&owner=exclude&action=getcompany'
    cik_re = re.compile(r'.*CIK=(\d{10}).*')

    cik_dict = {}
    return cik_re.findall(requests.get(url.format(ticker)).text)[0]

def test():
    import pandas as pd
    pd.core.common.is_list_like = pd.api.types.is_list_like
    from pandas_datareader import data as pdr
    import yfinance as yf
    yf.pdr_override()
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    TARKET_TICKER = 'CRM'
    get_read_form2 (TARKET_TICKER)
    data = pdr.get_data_yahoo(TARKET_TICKER, start="2017-01-01", end="2017-04-30")
    data['Adj Close'].plot()
    plt.show()

def RemoveNumericalTables(soup):
    '''
    Removes tables with >15% numerical characters.

    Parameters
    ----------
    soup : BeautifulSoup object
        Parsed result from BeautifulSoup.

    Returns
    -------
    soup : BeautifulSoup object
        Parsed result from BeautifulSoup
        with numerical tables removed.

    '''

    # Determines percentage of numerical characters
    # in a table
    def GetDigitPercentage(tablestring):
        if len(tablestring) > 0.0:
            numbers = sum([char.isdigit() for char in tablestring])
            length = len(tablestring)
            return numbers / length
        else:
            return 1

    # Evaluates numerical character % for each table
    # and removes the table if the percentage is > 15%
    [x.extract() for x in soup.find_all('table') if GetDigitPercentage(x.get_text()) > 0.15]

    return soup

def RemoveTags(soup):
    '''
    Drops HTML tags, newlines and unicode text from
    filing text.

    Parameters
    ----------
    soup : BeautifulSoup object
        Parsed result from BeautifulSoup.

    Returns
    -------
    text : str
        Filing text.

    '''
    import unicodedata

    # Remove HTML tags with get_text
    text = soup.get_text()

    # Remove newline characters
    text = text.replace('\n', ' ')

    # Replace unicode characters with their
    # "normal" representations
    text = unicodedata.normalize('NFKD', text)

    return text

def get_wordBox (URL):
    import urllib2
    from bs4 import BeautifulSoup
    import nltk
    from nltk.corpus import stopwords
    import re

    def cleanWord (word):
        text = re.sub ('[!@#$%^&*()_+-={}\|<>,.:;"]','',word)
        return text

    response3 = urllib2.urlopen(URL, timeout=6)

    #soup = BeautifulSoup(response3, 'lxml')
    soup = BeautifulSoup(response3, 'html5lib')

    docs = soup.find_all('document')
    words = docs[0].get_text().split()
    words = [cleanWord(word.lower()) for word in words]
    porter = nltk.PorterStemmer()
    stopWords = set(stopwords.words('english'))

    filter_words = []

#    print len (set(words))
    for word in set(words):
        if word not in stopWords:
            word = porter.stem(word)
            filter_words.append(word)

    #fdist = nltk.FreqDist(filter_words)
    #for word, frequency in fdist.most_common(25):
    #    print(u'{};{}'.format(word, frequency))

    return filter_words

def ComputeCosineSimilarity(words_A, words_B):
    '''
    Compute cosine similarity between document A and
    document B.

    Parameters
    ----------
    words_A : set
        Words in document A.
    words_B : set
        Words in document B

    Returns
    -------
    cosine_score : float
        Cosine similarity between document
        A and document B.

    '''
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # Compile complete set of words in A or B
    words = list(set(words_A) | set(words_B))

    # Determine which words are in A
    vector_A = [1 if x in words_A else 0 for x in words]

    # Determine which words are in B
    vector_B = [1 if x in words_B else 0 for x in words]

    # Compute cosine score using scikit-learn
    array_A = np.array(vector_A).reshape(1, -1)
    array_B = np.array(vector_B).reshape(1, -1)
    cosine_score = cosine_similarity(array_A, array_B)[0, 0]

    return cosine_score

def ComputeJaccardSimilarity(words_A, words_B):
    '''
    Compute Jaccard similarity between document A and
    document B.

    Parameters
    ----------
    words_A : set
        Words in document A.
    words_B : set
        Words in document B

    Returns
    -------
    jaccard_score : float
        Jaccard similarity between document
        A and document B.

    '''

    # Count number of words in both A and B
    words_intersect = len(list(set(words_A) & set (words_B)))

    # Count number of words in A or B
    words_union = len(list(set(words_A) | set(words_B)))

    # Compute Jaccard similarity score
    jaccard_score = words_intersect / words_union

    return jaccard_score

#words_A = get_wordBox ("https://www.sec.gov/Archives/edgar/data/1108524/0001108524-18-000087.txt")
#words_B = get_wordBox ("https://www.sec.gov/Archives/edgar/data/1108524/0001108524-18-000087.txt")

#print ComputeCosineSimilarity (words_A, words_B)
#print ComputeJaccardSimilarity (words_A, words_B)

dfFiling = get_read_form2('FB')

from tqdm import tqdm

#get_wordBox('https://www.sec.gov/Archives/edgar/data/1108524/0001108524-18-000011.txt')

#for idx in tqdm(range(0, len(dfFiling[['DATE_FILED','URL']])-1)):
for idx in range(0, len(dfFiling[['DATE_FILED','URL']])-1):
    last_date = dfFiling[['DATE_FILED','URL']].values[idx+1][0]
    words_A = get_wordBox(dfFiling[['DATE_FILED','URL']].values[idx][1])
    words_B = get_wordBox(dfFiling[['DATE_FILED','URL']].values[idx+1][1])
    print last_date, ComputeCosineSimilarity (words_A, words_B)
    #print last_date, ComputeJaccardSimilarity (words_A, words_B)


