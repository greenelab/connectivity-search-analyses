'''
This script is converts the adjacency.tsv file into an adjacency matrix that python/matlab can then compute paths from.
Outputs a txt file contaiing adjacen information stripped of all bio IDs,
and txt files containing the biological ID to numerial index conversion hashtables.
'''

import numpy as np
import sys
inputf = open('./adjacency/data-big/adjacency.tsv')
graphf = open('./adjacency/data-big/adj-list.tsv', 'w')


''' First get dictionary of ID names, and map to integers 1:N where N is number of disinct labels.
Example row of the tsv file:

Anatomy::UBERON:0000002	Disease::DOID:119	1 '''

ID2index = {}
index2ID = {}

num_names = 0

next(inputf) # skip header line
for line in inputf:
	line = line.strip()
	ID1, ID2, metaedge = line.split('\t')
	# convert object identifiers from strings to consecutive integers
	if ID1 in ID2index.keys():
		index1 = ID2index[ID1]
	else:
		ID2index[ID1] = num_names
		index2ID[num_names] = ID1
		index1 = num_names
		num_names = num_names + 1
	if ID2 in ID2index.keys():
		index2 = ID2index[ID2]
	else:
		ID2index[ID2] = num_names
		index2ID[num_names] = ID2
		index2 = num_names
		num_names = num_names + 1
	# Now num_names = number of distinct object identifiers, i.e. nodes
	graphf.write( str(index1) + '\t' + str(index2) + '\n' )

graphf.close()
inputf.close()

id2indexf = open('./adjacency/id2ind.tsv', 'w')
idlistf = open('./adjacency/ind2id.tsv', 'w')
print("Done reading in the graph\n")

for key in ID2index:
	id2indexf.write( key + '\t' + str(ID2index[key]) + '\n' )

for key in index2ID:
	idlistf.write( str(key) + '\t' + index2ID[key] + '\n' )

id2indexf.close()
idlistf.close()

