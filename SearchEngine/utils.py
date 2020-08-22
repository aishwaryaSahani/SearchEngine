'''
Created on April 28, 2020

@author: aishwaryaSahani
'''
# CS582: Information Retrieval
# University of Illinois at Chicago
# Spring 2020
# Semantic Search using Word Embeddings
# =========================================================================================================
import glob
import numpy as np
import string
import re
import spacy
spacy_nlp = spacy.load('en_core_web_sm')
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer 

# Function to calculate the cosine similarity between the query and document vector
# Arguments:
# doc: Represents the document vector of 300 dimensions
# query: Represents the query vector of 300 dimensions
# Returns: cosineSimilarity (float)
# Where, cosineSimilarity (float) fetches a float value between 0 to 1 to represent the similarity 
def get_cosine_similarity(doc, query):    
    return cosine_similarity(doc.reshape(1, -1), query.reshape(1, -1))[0][0]

# Function to clean the input text
# Arguments:
# tokens: A list of tokens that the document is split into
# Returns: cleanText (list)
# Where, cleanText (list) is a list of cleaned, preprocessed tokens 
def cleanText(tokens):
    cleanText=[]
    for word in tokens:
        if not (word.isdigit() or len(word)<3):
            cleanText.append(word)
    return cleanText

# Function to load the glove embeddings
# Returns: embeddings_dict (dict)
# Where, embeddings_dict (dict) is a dict of words as keys & the vector as values 
def load_glove():
    embeddings_dict= {}
    with open("Embeddings/glove.6B.300d.txt", 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict


# Function to load the documents and concatenate them to form a text(string)
# Arguments:
# tokens: A path of the input folder
# Returns: doc (list)
# Where, doc (list) is a list of string of words which appear in the documents
def loadfiles(folder):
    mylist = [f for f in glob.glob(folder+"/*")]
    doc =[]
    for file in sorted(mylist):
        text ="";
        f = open(file, "r")
        text+=(f.read())
        f.close()
        doc.append(text)
    return doc

# Function to read a text file
# Arguments:
# fileName: A string which has the name of the file in the relevant path
# Returns: text (string)
# Where, text (string) is a string of file data  
def readTextFile(fileName):
    f = open(fileName,"r")
    text = f.readlines()
    f.close()
    return text

# Function to split a document into a list of tokens
# Arguments:
# doc: A string containing input document
# Returns: tokens (list)
# Where, tokens (list) is a list of tokens that the document is split into
def get_tokens(doc):
    whiteSpaceToken = re.split("\\W+", doc)
    tokens = []
    transtable = str.maketrans('', '', string.punctuation)
    for word in whiteSpaceToken:
        tokens.append(word.lower().translate(transtable))
    return tokens


# Function to remove the stopwords & return tokens in the document
# Arguments:
# tokens: A list of tokens that the document is split into
# Returns: tokenList (list)
# Where, tokenList (list) is a list of non stop word tokens in the document
def removeStopWords(tokens):
    spacy_stopwords = spacy_nlp.Defaults.stop_words
    tokenList= []
    for word in tokens:
        if(word not in spacy_stopwords):
            tokenList.append(word)
    return tokenList 


# Function to lemmatize the list of tokens in the document
# Arguments:
# tokens: A list of tokens that the document is split into
# Returns: tokenList (list)
# Where, tokenList (list) is a list of tokens after lemmatizing in the document
def lemmatizer(tokens):
    lemmatizer = WordNetLemmatizer() 
    tokenList = [lemmatizer.lemmatize(token) for token in tokens]
    return tokenList

