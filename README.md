# READ_ME 
## Semantic Search using Word Embeddings
#### Name: Aishwarya Sahani
#### UIN: 652324475

The scope of the project would be to crawl & retrieve the webpages belonging “uic.edu” domain. These pages would be preprocessed. The tokenized words will be represented by word embeddings in terms of vectors. We will evaluate multiple word embeddings like word2vec, Glove to evaluate the best embedding for this purpose. These words will be combined with tf-idf values to represent the words. Using a cosine similarity metrics to evaluate the similarity of the document & selecting the top 10 based on their similarity. These pages would be then sorted using Pagerank algorithm to rank relevance,

The Search Engine would utilize & combine the information retrieval technique of tf-idf with a language modeling approach like word2vec/Glove to get the best results. In order to sort the results based on their importance, the pagerank algorithm has been incorporated for ranking the results.

### Steps to run:
1. Download the Crawled Pages Dataset from the link and save it in the project folder [Links](https://drive.google.com/file/d/115DwigYoGrdAtuodVFNOZlWfAC_Fl977/view?usp=sharing)
2. Download the 300 dimensional [Glove](http://nlp.stanford.edu/data/glove.6B.zip) embeddings
3. (Optional) You can download the 300 dimensional [word2vec](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download)  embeddings for comparison
4. Place the file in the Embeddings folder
5. You can run the searchEngine.py file 
6. The system will ask for a query. Input your query or press Enter to evaluate pre-defined queries
7. Wait for results. Check the file Output.txt for a summary of results.
8. To change the embeddings, uncomment the line which loads the embeddings in searchEngine.py

### Crawling the UIC (uic.edu) domain:
1. Run the file crawler.py.
2. You can set the min_url count to fetch the number of pages.
3. You can check the file Links.csv for results

### Libraries used:
- math:
 to perform mathematical operations like logarithms
- ast: 
 to use literal_eval to convert a string to an expression
- numpy:
 for large, multi-dimensional arrays and matrices, and array operations 
- pandas:
for data manipulation and analysis
- re:
for regular expression operation
- pkl:
handlign pickle files
- nltk:
Word Lemmatizer
- glob:
file retrieval 
- spacy:
for stop word list
- string:
for string punctuations during cleaning
- concurrent:
thread handling 
from concurrent.futures import ThreadPoolExecutor
- queue:
using data structure Queue
- requests:
request web pages
- urllib:
working with URLs
- bs4:
parsing HTML docs
- sys:
exception handling
- time:
for time handling & formatting
- gensim:
word2vec embeddings

### Contents:
The folder consists of 
1. Embeddings:
The folder consists of the 300 dimensional Glove embedding file and/or 300 dimensional word2vec embeddings
2. crawler.py
Run this file to crawl the web.
3. searchEngine.py
Run this file to run the Search Engine.
4. PageRank.py
The file contains the pagerank algorithm
5. utils.py
The file contains the utility methods used during the implementation
6. Links.csv
Contains the dataset of crawled pages
7. relevance.txt
Contains the gold standard of results for predefined queries
8. Output.txt
Contains the summary of results
9. Report.pdf
Contains the report of the project
