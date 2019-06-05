# created by Yun Hao @GreeneLab2019
#!/usr/bin/env python

import numpy as np
import pandas as pd
import requests

def calculate_edge_circa_score(query_drug, query_disease, query_tissue, circa_df, query_metapath = 'database', min_path_count = 1, amp_threshold = 0.1, fdr_threshold = 0.05):

	'''
	--Input  
	query_drug: drugbank ID
	query_disease: disease ontology ID
	query_tissue: list of tissue name (must match names used in CircaDB)
	circa_df: dataframe that contains CircaDB data ('data/circa_db_mapped.tsv')
	query_metapath: list of metapaths that will be included in the calculation (default only  includes all metapaths) 
	min_path_len: minimum number of paths bewteen query drug and query disease needed to generate a non-negative score (default 5) 
	amp_threshold: amplitude threshold to define circadian gene (default 0.1)
	fdr_threshold: FDR threshold to define circadian gene (default 0.05)
	
	-- Output:
	float: Edge circadian score
	String: reason why score cannot be calculated for query drug-disease pair (if any) 
	float: number of metapaths between drug-disease pair
	float: number of paths between drug-disease pair
	'''

	# number of query tissues
	tissue_len = len(query_tissue)
	
	# get hetionet ID of the drug and the disease
	drug_search = requests.get('https://search-api.het.io/v1/nodes/?search=' + query_drug).json()
	disease_search = requests.get('https://search-api.het.io/v1/nodes/?search=' + query_disease).json()

	# check whether the drug and the disease are in hetionet
	if len(drug_search['results']) == 0 and len(disease_search['results']) == 0:
		circa_edge_ratio = np.tile(float('nan'), tissue_len)
		total_meta_count = total_path_count = float('nan')
		note = 'query drug and disease not in hetionet'
	
	elif len(drug_search['results']) == 0:
		circa_edge_ratio = np.tile(float('nan'), tissue_len)
		total_meta_count = total_path_count = float('nan')
		note = 'query drug not in hetionet'
    	
	elif len(disease_search['results']) == 0:
		circa_edge_ratio = np.tile(float('nan'), tissue_len)
		total_meta_count = total_path_count = float('nan')
		note = 'query disease not in hetionet'
	
	else: 
		# get drug ID, disease ID, metapath ID
		drug_id = drug_search['results'][0]['id']
		disease_id = disease_search['results'][0]['id']
		gene_metapath = []
		# use metapaths in the database (sparse)
		if query_metapath == 'database':
			# get metapaths that connect the drug and the disease
			meta_search = requests.get('http://search-api.het.io/v1/query-metapaths/?source=' + str(drug_id) + '&target=' + str(disease_id)).json()
			if len(meta_search['path_counts']) > 0:
				for i in range(0, len(meta_search['path_counts'])):
					i_metapath = meta_search['path_counts'][i]['metapath_abbreviation']
					if 'G' in i_metapath:
						gene_metapath.append(i_metapath)
		# use pre-defined metapaths
		else:
			for i in range(0, len(query_metapath)):
				if 'G' in query_metapath[i]:
					gene_metapath.append(query_metapath[i])	
						
		# check whether drug and disease are connected by genes in hetionet
		if len(gene_metapath) == 0:
			circa_edge_ratio = np.tile(float('nan'), tissue_len)
			total_meta_count = total_path_count = float('nan')
			note = 'query drug and disease not connected by genes in hetionet'
			
		else:
			total_edge_score = np.zeros(tissue_len)
			circa_edge_score = np.zeros(tissue_len)
			total_meta_count = 0
			total_path_count = 0	
			query_amp = [x + '_amp' for x in query_tissue]
			query_fdr = [x + '_fdr' for x in query_tissue]
			for j in range(0, len(gene_metapath)):
				path_search = requests.get('http://search-api.het.io/v1/query-paths/?source=' + str(drug_id) + '&target=' + str(disease_id) + '&metapath=' + gene_metapath[j] + '&max-paths=-1').json()
				if len(path_search['paths']) == 0:
					continue

				elif path_search['query']['metapath_adjusted_p_value'] == 1:
					continue

				else:
					total_meta_count = total_meta_count + 1
					# get max circadian score of the path
					gene_path_loc = [int(index/2) for index, value in enumerate(gene_metapath[j]) if value == 'G']
					for k in range(0, len(path_search['paths'])):

						# get circadian score of the gene
						path_gene_circa_amp = np.zeros(tissue_len)
						path_gene_circa_fdr = np.zeros(tissue_len) + 1
						path_gene_circa_count = 0
						for l in range(0, len(gene_path_loc)):
							# get entrez ID of genes in the path
							loc_id = gene_path_loc[l]
							gene_node_id = path_search['paths'][k]['node_ids'][loc_id]
							gene_search =  requests.get('https://search-api.het.io/v1/nodes/' + str(gene_node_id)).json()
							gene_id = int(gene_search['identifier'])
							gene_circa_amp = circa_df[query_amp][circa_df['gene_id'] == gene_id]
							gene_circa_fdr = circa_df[query_fdr][circa_df['gene_id'] == gene_id]

							# check whether the gene is in circaDB
							if len(gene_circa_amp) == 0:
								continue
			
							else:
								path_gene_circa_count = path_gene_circa_count + 1
								for m in range(0, tissue_len):
									tmp_amp = float(gene_circa_amp.iloc[:,m])
									if tmp_amp > path_gene_circa_amp[m]:
										path_gene_circa_amp[m] = tmp_amp
										path_gene_circa_fdr[m] = float(gene_circa_fdr.iloc[:,m])
								
						if path_gene_circa_count > 0:
							# get path importance score
							total_path_count = total_path_count + 1
							path_score = path_search['paths'][k]['score']
							total_edge_score = total_edge_score + path_score
							# check whether the gene is circadian
							for n in range(0, tissue_len):
								if path_gene_circa_amp[n] >= amp_threshold and path_gene_circa_fdr[n] < fdr_threshold:
									circa_edge_score[n] = circa_edge_score[n] + path_score

			# calculate proportion of circadian paths
			if total_meta_count == 0:
				circa_edge_ratio = np.tile(float('nan'), tissue_len)
				total_path_count = float('nan')
				note = 'query drug and disease not connected by genes in hetionet'
			else:
				if total_path_count == 0:
					circa_edge_ratio = np.tile(float('nan'), tissue_len)
					note = 'query drug and disease connected by genes not in CircaDB'
				elif total_path_count < min_path_count:
					circa_edge_ratio = np.tile(float('nan'), tissue_len)
					note = 'query drug and disease connected by too few paths'
				else:
					circa_edge_ratio = circa_edge_score/total_edge_score
					note = 'NaN'

	return circa_edge_ratio, note, total_meta_count, total_path_count
