'''
Created on April 28, 2020

@author: aishwaryaSahani
'''
# CS582: Information Retrieval
# University of Illinois at Chicago
# Spring 2020
# Semantic Search using Word Embeddings
# =========================================================================================================
import utils
import numpy as np
import pandas as pd
import math
import PageRank as pr
from ast import literal_eval
import gensim
 
docCounterList = []


def computeWordVectorQuery(doc, embeddings_dict, tfidf):
    vector = np.zeros(300)
    count = 0
    for word in doc:
        if word in embeddings_dict:
            vector += embeddings_dict[word] * tfidf[word]
#         else:
#             vector += tfidf[word]*np.ones(300)
            count = count +1
    if(count>0):
        vector = vector/count
    return vector

def computeWordVector(doc, embeddings_dict, tfidf, index):
    vector = np.zeros(300)
    count = 0
    for word in doc:
        if word in embeddings_dict:
            # combining tfidf & word embeddings by dot product
            vector += embeddings_dict[word] * tfidf[word][index]
#         else:
#             vector += tfidf[word][index]*np.ones(300)
            count = count +1
    if(count>0):
        vector = vector/count
    return vector

# Function to preprocess the raw documents & produce clean words as output
# Arguments:
# rawDocs: A list of tokens of rawDocs that the document is split into
# isSGML: A boolean variable to identify SGML tags 
# Returns: cleanWords (list)
# Where, cleanWords (list) is a list of list of tokens in the documents after preprocessing 
def preprocess(doc, query):
    if query:
        tokens = utils.get_tokens(doc)
        doc = utils.removeStopWords(tokens)
    lemmaWords = utils.lemmatizer(doc)
    cleanWords = utils.cleanText(lemmaWords)
    return cleanWords

# Function to calculate the inverted index as output from the list of list of tokens of docs
# Arguments:
# docs: A list of tokens of rawDocs that the documents are split into
# Returns: invertedIndex (dict), maxElement (list)
# Where, invertedIndex (dict) is a dict of dict of terms as keys and documents and their count as values
# maxElement (list) is a list of max count of words in each doc
def calculateInvertedIndex(docs):
    invertedIndex = {}
    docCounter =0
    maxElement = {}
    for doc in docs:
        max =0
        for word in doc:
            if word not in invertedIndex:
                invertedIndex[word] = {}
                invertedIndex[word][docCounter] = 1
            else:
                if docCounter in invertedIndex[word]: 
                    invertedIndex[word][docCounter] = invertedIndex[word][docCounter]+1
                else:
                    invertedIndex[word][docCounter] = 1
            if invertedIndex[word][docCounter] > max:
                max =  invertedIndex[word][docCounter]
        
        maxElement[docCounter] = max
        docCounter+=1
    return invertedIndex, maxElement  

# Function to calculate the TFIDF from inverted index
# Arguments:
# invertedIndex: A dict of inverted Index
# maxElement: A list of max count of words in each doc
# N: int value of number of docs in dataset
# Returns: tfidf (dict)
# Where, tfidf (dict) is a dict of dict of terms as keys and tfidf corresponding to docs as values
def calculateTFIDF(invertedIndex, maxElement, N):
    tfidf = invertedIndex
    for word in tfidf:
        df = len(tfidf[word])
        for doc in tfidf[word]:
            tfidf[word][doc] = tfidf[word][doc]/maxElement[doc] * math.log(N/df,2)
    return tfidf  

# Function to calculate the TFIDF from inverted index for query
# Arguments:
# invertedIndex: A dict of inverted Index
# max: An int count of max occurences
# N: int value of number of docs in dataset
# tfidf: dict of dict of tfidf
# Returns: tfidf (dict)
# Where, weights (dict) is a dict of dict of terms as keys and tfidf corresponding to docs as values
def calculateTFIDFQuery(invertedIndex, max, tfidf, N):
    weights = invertedIndex
    for word in weights:
        if word in tfidf:
            df = len(tfidf[word])
            weights[word] = weights[word]/max * math.log(N/df,2)
    return weights  

# Function to calculate the inverted index as output from query
# Arguments:
# query: A list of tokens of the query
# Returns: invertedIndex (dict), max (int)
# Where, invertedIndex (dict) is a dict of dict of terms as keys and their count as values
# maxElement (list) is a list of max count of words 
def calculateInvertedIndexQuery(query):
    invertedIndex = {}
    max = 0
    for word in query:
        if word not in invertedIndex:
            invertedIndex[word] = 1
        else:
            invertedIndex[word] = invertedIndex[word]+1
        if invertedIndex[word] > max:
            max =  invertedIndex[word]
    return invertedIndex, max  

# Function to combine the TFIDF with the avg word embeddings of the documents   
# Arguments:
# embeddings_dict: A dict of word embeddings with words as keys and embeddings as values
# docs: The list of doc as tokens 
# query: The query as tokens
# tfidf: dict of dict of tfidf of docs
# weightsQuery: dict of dict of tfidf of query
# Returns: tfidf (dict)
# Where, cosineValues (list) is a list of cosine values of pages 
def embeddingsResult(embeddings_dict, docs, query, tfidf, weightsQuery):
     
    queryVector = computeWordVectorQuery(query, embeddings_dict, weightsQuery) 
    cosineValues = {}
    for index,doc in enumerate(docs):
        vector = computeWordVector(doc, embeddings_dict, tfidf, index)
        cosineValues[index] = utils.get_cosine_similarity(vector, queryVector)
    return cosineValues

# Function to calculate the precision & recall for N docs
# Arguments:
# N: Total number of docs for consideration
# relDict: A dict with key as query and values as relevance docs
# cumulativeCosSims: dict of sorted cosine Similarity values in descending order 
def computePrecisionRecallforNDoc(N,relDict, outputDict):
    f = open("Output.txt", "w")
    print("Top "+str(N)+" documents in rank list")
    f.write("Top "+str(N)+" documents in rank list\n")
    avg_pr = 0
    avg_re= 0
    for query, doclist in outputDict.items():
        tp = 0
        for doc in doclist[:N]:
            if doc in relDict[query+1]:
                tp +=1
        pr = tp/N
        re = tp/len(relDict[query+1])
        f1 = 2*pr*re/(pr+re)
        avg_pr += pr
        avg_re += re
        print("Query: "+str(query+1)+"\t Pr: "+str(pr)+"\t Re: "+str(re)+"\t F1: "+str(f1))
        f.write("Query: "+str(query+1)+"\t Pr: "+str(pr)+"\t Re: "+str(re)+"\t F1: "+str(f1))
    
    avg_pr = avg_pr/len(relDict)
    avg_re= avg_re/len(relDict)
    avg_f1 = 2*avg_pr*avg_re/(avg_pr+avg_re)    
    print("Avg Precision:"+str(avg_pr))
    f.write("Avg Precision:"+str(avg_pr)+"\n")
    print("Avg Recall:"+str(avg_re))
    f.write("Avg Recall:"+str(avg_re)+"\n")
    print("Macro Avg F1:"+str(avg_f1))
    f.write("Macro Avg F1:"+str(avg_f1)+"\n")
    print()
    f.write("\n")

# Function to calculate the relevant docs for each query
# Arguments:
# relevance: The text string of file contents of relevance
# Returns: relDict (dict)
# Where, relDict (dict) is a dict of query as key and relevant docs as values    
def relevanceDict(relevance):
    relDict = {}
    for line in relevance:
        query,doc = line.split()
        query = int(query)
        if query not in relDict:
            documents = []
            documents.append(doc)
            relDict[query] = documents
        else:
            documents.append(doc)
            relDict[query] = documents
    return relDict

# Function to load the links data
# Returns: cleanWords (list) , linksDocs (dict)
# Where, cleanWords (list) is a list of tokens of preprocessed documents
# Where, linksDocs (dict) is a dict with the url as keys and the referenced urls as a list as values 
def loadData():
    webPagesDF = pd.read_csv("Links.csv")
    cleanWords = []
    linksDocs = {}
    print("Loading crawled pages")
    for index,row in webPagesDF.iterrows():
        if row["text"] is not None and row["text"] != "" and type(row["text"]) is str:
            if row["page"] not in docCounterList:
                tokens = literal_eval(row["text"])
                words = preprocess(tokens, False)
                docCounterList.append(row["page"])
                cleanWords.append(words)
                linksDocs[row["page"]] = set(literal_eval(row["link"]))
    return cleanWords, linksDocs

# Function to search the data
# Arguments:
# queries: A list of queries as text strings 
# embeddings_dict: A dict of word embeddings with words as keys and embeddings as values
# cleanWords: A list of tokens of preprocessed documents
# tfidf: dict of dict of tfidf of docs
# Where, outputDict (dict) is a dict with the query as keys and the correponding searched urls as a list as values 
def search(queries, embeddings_dict, cleanWords, tfidf, userQuery):
    
    outputDict = {}
    queryCounter = 0
    
    pageSize = 10 if userQuery is False else 50 

    for query in queries:
        print("\nThe results of the query:")
        print(query+"\n")
        
        # preprocess & calculate invertedindex of the query & then calculate tfidf
        cleanQuery = preprocess(query, True)
        invertedIndexQuery, maxQuery = calculateInvertedIndexQuery(cleanQuery)
        weightsQuery = calculateTFIDFQuery(invertedIndexQuery, maxQuery, tfidf, N)
        
        #merge tfidf values with embeddings and calculate cosine similarity
        embedResults = embeddingsResult(embeddings_dict, cleanWords, cleanQuery, tfidf, weightsQuery)
        sortedCS = sorted(embedResults.items(), key=lambda x: x[1],reverse=True)
        
        output = (list(dict(sortedCS).keys())[:pageSize])
        offset =0
        while(True):
            print()
            outputList = []
            pageRankDict = {}
            for doc in output[offset:offset+10]:
                outputList.append(docCounterList[doc])
                pageRankDict[docCounterList[doc]] = pageranks[docCounterList[doc]]
            outputDict[queryCounter] = outputList
            
            sortedoutput = sorted(pageRankDict.items(), key=lambda x: x[1],reverse=True)  

            for doc in sortedoutput:
                print(doc[0]) 
            
            if userQuery:
                cont = input("Do you want more results? Press Y/N")
                if((cont!='Y' and cont!='y')):
                    break
                if (offset==pageSize-10):
                    print("All relevant pages fetched")
                    break
                offset += 10
            else:
                break
        queryCounter+=1
        
    query = input("Do you want to exit? Press Y otherwise enter query?")
    if query != "Y" and query!="y":
        queries = [query]
        userQuery =True  
        outputDict = search(queries, embeddings_dict, cleanWords, tfidf, userQuery)
        
    return outputDict, userQuery

if __name__ == "__main__":

    # load crawled pages
    cleanWords, linksDocs = loadData()
    N = len(cleanWords)  

    # calculate the inverted index & the tf idf
    print("Loading inverted index")
    invertedIndex, maxCount = calculateInvertedIndex(cleanWords)
    print("Loading tf-idf")
    tfidf = calculateTFIDF(invertedIndex, maxCount, N )
    
#     doIndexingTfIDF(tfidf)
#     doIndexingPreprocessedWord()
#     doIndexingDocuments()
#     doIndexingPR()
    
#     tfidf = fetchIndexingTfIDF(tfidf)
#     cleanWords = fetchIndexingPreprocessedWord()
#     docCounterList = fetchIndexingDocuments()
#     pageranks = fetchIndexingPR()
        
    # calculate pagerank
    print("Loading pagerank")
    pageranks = pr.compute(linksDocs)
    
    # load any embeddings of choice
    print("Loading word embeddings")
    embeddings_dict =  utils.load_glove()
#     embeddings_dict = gensim.models.KeyedVectors.load_word2vec_format('Embeddings\GoogleNews-vectors-negative300.bin', binary=True)    
    
    #list of predefined queries
    queries = ["Professors who teach NLP at UIC","Student organizations at uic", "Student Orientation at UIC","How has coronavirus affected UIC", "Centers for Cultural Understanding and Social Change"]
    # press enter to validate pre-defined queries 
    # otherwise input your query
    query = input("Please enter your query or Press enter to evaluate pre-defined queries")
    userQuery =False
    if query != "":
        queries = [query]
        userQuery =True  
    
    outputDict, userQuery = search(queries, embeddings_dict, cleanWords, tfidf, userQuery)
    
    embeddings_dict = {}
    
    if userQuery is False:
        # map with gold standard
        relevance = utils.readTextFile("relevance.txt")
        relDict = relevanceDict(relevance)
        
        # compute the precision & recall values
        computePrecisionRecallforNDoc(5, relDict, outputDict)
        computePrecisionRecallforNDoc(10, relDict, outputDict)