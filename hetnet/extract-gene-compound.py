''' This script is to extract the gene and compound IDs and map to their ajacency list indexes.'''

import numpy as np
import sys

id2indexf = open('./adjacency/id2ind.tsv','r')

''' 
Transform the files into dictionaries that map to and from node ID to index.
Examples of node IDs:
Gene::
Compound::
Anatomy::UBERON:0000002
Disease::DOID:119
'''

# Construct hashtable from file
ID2index = {}
index2ID = {}

for line in id2indexf:
	line = line.strip()
	ID, index = line.split('\t')
	ID2index[ID] = int(index)
	index2ID[int(index)] = ID

id2indexf.close()

print("Done reading in hashtable.\n")

# Construct list of gene and compound nodes
GeneIndexes = []
CompoundIndexes = []
num_genes = 0
num_compounds = 0
for key in ID2index:
	if "Gene::" in key:
		GeneIndexes.append(ID2index[key])
		num_genes = num_genes + 1
	if "Compound::" in key:
		CompoundIndexes.append(ID2index[key])
		num_compounds = num_compounds + 1

print(str(num_genes))
print(str(num_compounds))

# Write gene and compound node lists to file
genelistf = open('./adjacency/genelist.tsv','w')
compoundlistf = open('./adjacency/compoundlist.tsv','w')
for index in GeneIndexes:
	genelistf.write( str(index) + '\t' + index2ID[index] + '\n' )
genelistf.close()
for index in CompoundIndexes:
	compoundlistf.write( str(index) + '\t' + index2ID[index] + '\n' )
compoundlistf.close()

print("Gene and compound index lists done.\n")

