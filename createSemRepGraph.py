import graph_tool
import pandas as pn
import graph_tool.util
import graph_tool.topology
import graph_tool.centrality
import graph_tool.stats
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from xml.etree import cElementTree as elementTree
import re
import graph_tool
import graph_tool.util
import graph_tool.stats
from numpy.random import *  # for random sampling


############################################
# small utility functions
############################################
def getPathMaxPageRank(path):
    pr = []
    for p in path:
        pr.append(pagerank[p])
    return max(pr)


def getPathNames(path):
    pathStrArry = []
    for p in path:
        txt = IdToStrDict[p]
        pathStrArry.append(txt)
    return pathStrArry


def printPageRankHist(semRepGraph):
    prarry = []
    for n in semRepGraph.nodes():
        prarry.append(pagerank[n])


semRepGraph = graph_tool.Graph()
text_property = semRepGraph.new_vertex_property("string")
semRepGraph.vertex_properties['text'] = text_property

semType_property = semRepGraph.new_vertex_property("string")
semRepGraph.vertex_properties['semType'] = semType_property

cui_property = semRepGraph.new_vertex_property("string")
semRepGraph.vertex_properties['cui'] = cui_property

predicates = pn.read_csv('/my_folder/SemRep.txt')  # test2.txt is from SemRep
predicates.columns = ['PREDICATION_ID', 'SENTENCE_ID', 'PMID', 'PREDICATE', 'SUBJECT_CUI', 'SUBJECT_NAME',
                      'SUBJECT_SEMTYPE',
                      'SUBJECT_NOVELTY', 'OBJECT_CUI', 'OBJECT_NAME', 'OBJECY_SEMTYPE', 'OBJECT_NOVELTY']

'''
Filter data by Predicate type.
In this example, we leave only 'CAUSES'
'''
predTypes = ['CAUSES']
predicates = predicates[predicates['PREDICATE'].isin(predTypes)]

'''
Data can also be filtered by OBJECT_SEMTYPE and SUBJECT_SEMTYPE.
For example:
semtypes=['orgm','bpoc','diap','ortf','bsoj','inpo','tisu','topp','mamm','inpr','geoa','hlca','bdsy','blor','hcro','lbpr'
          ,'inbe','orga','menp','mnob','humn','amph','plnt','spco','anim','resa','anab','eehu','tmco','edac','ftcn','ocdi',
          'dora','qnco','orgt','npop','qlco','podg','prog','bird','mcha','rept','fish','phob','socb','idcn','popg',
          'bmod','emod','aggp','famg','rnlw','mbrt','pros','lang','ocac','gora','medd']

predicates = predicates[predicates['OBJECT_SEMTYPE'].isin(semtypes)==False]
predicates = predicates[predicates['SUBJECT_SEMTYPE'].isin(semtypes)==False]

'''

nodeDict = {}

subjects = predicates['SUBJECT_NAME'].unique()
objects = predicates['OBJECT_NAME'].unique()
nodes = list(set(subjects).union(set(objects)))
for n in nodes:
    newNode = semRepGraph.add_vertex()
    text_property[newNode] = n
    nodeDict[n] = int(newNode)
'''
subjectcuidf = predicates[['SUBJECT_NAME', 'SUBJECT_CUI']].copy()
subjectcuidf = subjectcuidf.drop_duplicates()
subjectcuidf = subjectcuidf.rename(columns={'SUBJECT_NAME': 'NAME', 'SUBJECT_CUI': 'CUI'})

objectcuidf = predicates[['OBJECT_NAME', 'OBJECT_CUI']].copy()
objectcuidf = objectcuidf.drop_duplicates()
objectcuidf = objectcuidf.rename(columns={'OBJECT_NAME': 'NAME', 'OBJECT_CUI': 'CUI'})

cuidf = subjectcuidf.append(objectcuidf)
'''

index = 0
edges = []
while index < len(predicates):
    print(index)
    object = nodeDict[predicates.iloc[index]['SUBJECT_NAME']]
    subject = nodeDict[predicates.iloc[index]['OBJECT_NAME']]
    edges.append((object, subject))
    index += 1

semRepGraph.add_edge_list(edges)

graph_tool.stats.remove_self_loops(semRepGraph)

semRepGraph.save("/my_folder/semRepGraph.xml")  # save garph
save_obj(nodeDict, "/my_folder/nodeDict")  # save dictionary

# load graph from memory
semRepGraph = graph_tool.load_graph("/my_folder/semRepGraph.xml")
nodeDict = load_obj("/my_folder/nodeDict")

# init IdTpStrDict
IdToStrDict = {}
for k in nodeDict.keys():
    IdToStrDict[nodeDict[k]] = k

# calculate page rank for path filtering. Example of filtering by max page rank shown below
pagerank = graph_tool.centrality.pagerank(semRepGraph)

# get source and target nodes according to the object of subject name in UMLS

s = nodeDict['Hydroxymethylglutaryl-CoA Reductase Inhibitors']
t = nodeDict['Blood Pressure']

# find paths from source to target
paths = graph_tool.topology.all_paths(semRepGraph, s, t, cutoff=2)

# convert paths to text paths (paths where nodes have UMLS names)
count = 0
textpaths = pn.DataFrame()
prarray = []  # will hold max page rank for each path
for p in paths:
    prarray.append(getPathMaxPageRank(p))
    textpaths = textpaths.append(pn.DataFrame([getPathNames(p)]))
    count += 1
    print(count)
textpaths = textpaths.drop_duplicates()

print(len(textpaths))

