'''
This script is converts the adjacency.tsv file into an adjacency matrix that python/matlab can then compute paths from.
Outputs a txt file containing adjacency information stripped of all bio IDs,
and txt files containing the biological ID to numerial index conversion hashtables.
Also extracts the gene and compound IDs and write to file a hashmap from their node IDs to node indexes.
'''
import numpy as np
import sys

inputf = open('./adjacency/data-big/adjacency.tsv')
graphf = open('./adjacency/data-big/adj-list.tsv', 'w')
graphf.write('int_id\tint_id\n') # Write header info into the adj-list

# First get dictionary of ID names, and map to integers 1:N where N is number of disinct labels.
ID2index = {}
num_names = 0

next(inputf) # skip header line
for line in inputf:
	line = line.strip()
	ID1, ID2, metaedge = line.split('\t')
	# convert object identifiers from strings to consecutive integers
	if ID1 in ID2index:
		index1 = ID2index[ID1]
	else:
		ID2index[ID1] = num_names
		index1 = num_names
		num_names = num_names + 1
	if ID2 in ID2index:
		index2 = ID2index[ID2]
	else:
		ID2index[ID2] = num_names
		index2 = num_names
		num_names = num_names + 1
	# Now num_names = number of distinct object identifiers, i.e. nodes
	graphf.write( str(index1) + '\t' + str(index2) + '\n' )

graphf.close()
inputf.close()
print("Done reading in the graph\n")


with open('./adjacency/ind2id.tsv', 'w') as ind2idf:
	ind2idf.write('int_id\tnode_id\n') # Write header info
	for key in ID2index:
		ind2idf.write( str(ID2index[key]) + '\t' + key + '\n' )

# Next, extract gene and compound ids
with open('./adjacency/ind2id.tsv','r') as index2idf:
	next(index2idf) # skip header line
	ID2index = {}
	index2ID = {}
	for line in index2idf:
		line = line.strip()
		index, ID = line.split('\t')
		ID2index[ID] = int(index)
		index2ID[int(index)] = ID

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
	genelistf.write( 'int_id\tnode_id\n' ) # Write header info
	for index in GeneIndexes:
		genelistf.write( str(index) + '\t' + index2ID[index] + '\n' )

with open('./adjacency/compoundlist.tsv','w') as compoundlistf:
	compoundlistf.write( 'int_id\tnode_id\n' ) # Write header info
	for index in CompoundIndexes:
		compoundlistf.write( str(index) + '\t' + index2ID[index] + '\n' )

print("Gene and compound index lists done.\n")


