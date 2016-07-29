''' This script loads the adjacency information from ./data-big and constructs a sparse matrix from it.
Also loads the gene and compound ID lists and generates hashtables to map from Matrix index to gene and compound ID.'''

import numpy as np
import scipy as sp
import networkx as nx
import sys
import os
import scipy.sparse

with open("./adjacency/data-big/adj-list.tsv", 'r') as graphfile:
	next(graphfile)
	G=nx.read_edgelist(graphfile, nodetype=int)
smat = nx.to_scipy_sparse_matrix(G)
print ("Done reading in graph and converting to a matrix.\n")

# Now load gene and compound node lists
with open('./adjacency/genelist.tsv','r') as genelistf:
	next(genelistf) # skip header line
	GeneIndexes = []
	GeneNames = {}
	GeneName2index = {}
	for line in genelistf:
		a, b = line.split('\t')
		GeneIndexes.append(int(a))
		GeneNames[int(a)] = b.strip()
		GeneName2index[b.strip()] = int(a)

with open('./adjacency/compoundlist.tsv','r') as compoundlistf:
	next(compoundlistf) # skip header line
	CompoundIndexes = []
	CompoundNames = {}
	CompoundName2index = {}
	for line in compoundlistf:
		a, b = line.split('\t')
		CompoundIndexes.append(int(a))
		CompoundNames[int(a)] = b.strip()
		CompoundName2index[b.strip()] = int(a)

print("Done loading genes and compounds.\n")

# Load node ID index map
index2ID = {}
with open('./adjacency/ind2id.tsv', 'r') as idlistf:
	next(idlistf) # skip header line
	for line in idlistf:
		line = line.strip()
		index, ID = line.split('\t')
		index2ID[int(index)] = ID

print("Done loading node ID-index map.\n")
