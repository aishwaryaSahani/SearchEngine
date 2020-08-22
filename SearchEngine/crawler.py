'''
Created on April 28, 2020
 
@author: aishwaryaSahani
'''
#  CS582: Information Retrieval
#  University of Illinois at Chicago
#  Spring 2020
#  Semantic Search using Word Embeddings
#  =========================================================================================================
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import pandas as pd
import requests
from urllib.request import urlparse, urljoin 
from bs4 import BeautifulSoup
import time
import sys
import utils

pagesList= []
urls_visited = set()
urls_queue = Queue()
urls_list = []
min_urls = 3000
df = pd.DataFrame(columns=["page","text","link"])
pool = ThreadPoolExecutor(max_workers=10)

# Function to preprocess the documents to return normalized text
# Arguments:
# doc: A document in text string
# Returns: cleanWords (list)
# Where, cleanWords (list) is a list of normalized tokens in the document after preprocessing
def preprocess(doc):
    tokens = utils.get_tokens(doc)
    tokensWOStopWords = utils.removeStopWords(tokens)
    cleanWords = utils.cleanText(tokensWOStopWords)
    return cleanWords


# Function to check if the url string is valid 
# Arguments:
# url: The url as text string
# Returns: validity (boolean)
# Where, validity (boolean) is a boolean value based on the validity of the url
def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)
 
# Function to retrieve the hyperlinks referenced in the page
# Arguments:
# text: The text content of the page as text string
# url: The url of the page as text string
def get_links(text, url):
    time.sleep(1)
    content = {}
    linkList = []
    soup = BeautifulSoup(text, "html.parser")
    links = soup.findAll("a", href=True)
    for link in links:
        href = link['href']
        href = urljoin(url, href)
        parsed_href = urlparse(href)
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
         
        if href.endswith("/"):
            href = href[:-1]
         
        if "https" not in href:
            href = href.replace("http","https")
         
        if href not in urls_list:
            if "uic.edu" in href and "@" not in href and is_valid(href):
                urls_queue.put(href)
                urls_list.append(href)
                linkList.append(href)
        else:
            linkList.append(href)
    
    # remove JS & CSS content
    for script in soup(["script", "style"]):
        script.extract()  
    
    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    doc = preprocess(text)  
    
    if url.endswith("/"):
            url = url[:-1]
         
    if "https" not in url:
        url = url.replace("http","https")

    content["page"] = url
    content["text"] =  doc
    content["link"] = linkList
    pagesList.append(content)
    
    return 

# Function to request the page from the url
# Arguments:
# url: The url of the page as text string
# Returns: res (response)
# Where, res (response) is the response after the page is requested
def scrape_page(url):
    try:
        res = requests.get(url)
        return res
    except requests.RequestException:
        return

# Post Callback Function to get links after successful page load
# Arguments:
# res: the response after the page is requested
def post_scrape_callback(res):
    result = res.result()
    if result and result.status_code == 200:
        get_links(result.text, result.url)


# Function to get crawl all the pages
def crawl():  
    
    while len(pagesList)<min_urls:
        try:
            target_url = urls_queue.get(timeout=60)
            if target_url not in urls_visited:
                print("Scraping URL: {}".format(target_url))
                # request the page
                job = pool.submit(scrape_page, target_url)
                # request links of the page
                job.add_done_callback(post_scrape_callback)
                urls_visited.add(target_url)
                if len(pagesList)%(min_urls*0.1) == 0:
                    print("Crawling status = "+str(len(pagesList)*100/min_urls)+"%")
                
        except Exception as e:
            print(e)
            continue
         
if __name__ == "__main__":
    
    # use a queue to initialize crawling
    seedURL = "https://cs.uic.edu"
    urls_queue.put(seedURL)
    urls_list.append(seedURL)
    print("Crawling started")
    start_time = time.time()
    try:
        crawl()
    except:
        e = sys.exc_info()[0]
        print( "<p>Error: %s</p>" % e )
        
    time.sleep(100)
    df = df.append(pagesList)
    df.to_csv("Links.csv")
    
    print("Crawling completed")
    print("Crawlingtime - %s seconds" % (time.time() - start_time))
    print("Collected URLs:", len(pagesList))
    
    sys.exit()

