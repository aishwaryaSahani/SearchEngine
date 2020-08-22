'''
Created on April 10, 2020

@author: aishwaryaSahani
'''
import numpy as np

# Function to create a matrix from the graph of words 
# Arguments:
# graph: A dict of list where words are keys & the values are the adjacent/neighboring words
# Returns: matrix (2d numpy array), coveredNodeList(list)
# Where, matrix (2d numpy array) is a matrix with the row & columns as the graph nodes & the values representing the edge weight
# Where, coveredNodeList (list) is a list of nodes in the graph
def createMatrix(graph):
    matrix = np.zeros((len(graph.keys()), len(graph.keys())), dtype='float')
    coveredNodeList = list(graph.keys())
    i = 0
    for node, edges in graph.items():
        for edge in edges:
            if edge in coveredNodeList:
                j = coveredNodeList.index(edge)
                matrix[i][j] += 1/len(edges)
        i += 1
    return  matrix, coveredNodeList

# Function to calculate the pagerank of the pages from the matrix
# Arguments:
# matrix: A matrix with the row & columns as the graph nodes & the values representing the edge weight
# Returns: S (numpy array)
# Where, S (numpy array) is a numpy array with the pagerank of  pages from the matrix 
def calculatePageRank(matrix):
    alpha = 0.85
    p = [1/matrix.shape[0]]*matrix.shape[0]
    p = np.array(p) 
    S = [1/matrix.shape[0]]*matrix.shape[0]
    S = np.array(S)
    for i in range(10):
        S = (alpha)*np.dot(S, matrix) + (1-alpha)*p
    return S

# Function to calculate the pagerank
# Arguments:
# Returns: pr (dict)
# Returns: coveredNodeList (list)
# Where, pr (dict) is a dict of pages and their pagerank values as key, value 
def compute(linksDocs): 
    matrix, coveredNodeList = createMatrix(linksDocs)
    # calculate PageRank
    pagerank = calculatePageRank(matrix)
    
    pr = {}
    counter = 0
    for doc in coveredNodeList:
        pr[doc] = pagerank[counter]
        counter+=1
        
    return pr
    