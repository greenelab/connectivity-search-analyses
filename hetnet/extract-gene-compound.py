''' This script is to extract the gene and compound IDs and map to their ajacency list indexes.'''

import numpy as np
import sys

# Construct node-ID maps from file
with open('./adjacency/ind2id.tsv','r') as index2idf:
	next(index2idf) # skip header line
	ID2index = {}
	index2ID = {}
	for line in index2idf:
		line = line.strip()
		index, ID = line.split('\t')
		ID2index[ID] = int(index)
		index2ID[int(index)] = ID

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

print(str(num_genes) + ' genes')
print(str(num_compounds) + ' compounds')

# Write gene and compound node lists to file
with open('./adjacency/genelist.tsv','w') as genelistf:
	genelistf.write( '# genelist: int_id \t node_id \n' ) # Write header info
	for index in GeneIndexes:
		genelistf.write( str(index) + '\t' + index2ID[index] + '\n' )

with open('./adjacency/compoundlist.tsv','w') as compoundlistf:
	compoundlistf.write( '# compoundlist: int_id \t node_id \n' ) # Write header info
	for index in CompoundIndexes:
		compoundlistf.write( str(index) + '\t' + index2ID[index] + '\n' )

print("Gene and compound index lists done.\n")

