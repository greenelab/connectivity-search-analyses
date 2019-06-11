# created by Yun Hao @GreeneLab2019
#!/usr/bin/env python

import numpy as np
import pandas as pd
import requests

## This function calculates the edge circadian score of a drug-disease pair without returning the score details of each path. It is faster, and requires less memory
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
	list: Edge circadian score
	String: reason why score cannot be calculated for query drug-disease pair (if any) 
	float: number of metapaths that contain gene
	float: number of paths that contain gene in CircaDB
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
							gene_id = int(path_search['nodes'][str(gene_node_id)]['properties']['identifier'])
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


## This function calculates the edge circadian score of a drug-disease pair, and then return the score details of each path. It is slower, and requires more memory
def detail_edge_circa_score(query_drug, query_disease, query_tissue, circa_df, query_metapath = 'database', min_path_count = 1, amp_threshold = 0.1, fdr_threshold = 0.05):

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
	
	-- Output
	dictionary contains the following item:
		edge_circa_score
		total_meta_count: number of metapaths that contain gene
		total_path_count: number of paths that contain gene in CircaDB
		note: reason why score cannot be calculated for query drug-disease pair (if any) 
		score_details: a dataframe that contains score details of each path
	'''

	# number of query tissues
	tissue_len = len(query_tissue)
	
	# get hetionet ID of the drug and the disease
	drug_search = requests.get('https://search-api.het.io/v1/nodes/?search=' + query_drug).json()
	disease_search = requests.get('https://search-api.het.io/v1/nodes/?search=' + query_disease).json()
	null_output = {'edge_circa_score': dict((query_tissue[x], float('nan')) for x in range(0, len(query_tissue))),
			'total_meta_count': float('nan'),
			'total_path_count': float('nan'),
			'note': 'NaN',
			'score_details': 'NaN'
			}

	# check whether the drug and the disease are in hetionet
	if len(drug_search['results']) == 0 and len(disease_search['results']) == 0:
		output = null_output
		output['note'] = 'query drug and disease not in hetionet'		
	
	elif len(drug_search['results']) == 0:
		output = null_output
		output['note'] = 'query drug not in hetionet'
	
	elif len(disease_search['results']) == 0:
		output = null_output
		output['note'] = 'query disease not in hetionet'
	
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
			output = null_output
			output['note'] = 'query drug and disease not connected by genes in hetionet'
			
		else:
			total_edge_score = np.zeros(tissue_len)
			circa_edge_score = np.zeros(tissue_len)
			total_meta_count = 0
			total_path_count = 0
			score_details = []
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
						path_gene_name = []
						for l in range(0, len(gene_path_loc)):
							# get entrez ID of genes in the path
							loc_id = gene_path_loc[l]
							gene_node_id = path_search['paths'][k]['node_ids'][loc_id]
							gene_id = int(path_search['nodes'][str(gene_node_id)]['properties']['identifier'])
							gene_name = path_search['nodes'][str(gene_node_id)]['properties']['name']
							path_gene_name.append(gene_name)
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
										tmp_tissue = query_tissue[m]
										path_gene_circa_amp[m] = tmp_amp
										path_gene_circa_fdr[m] = float(gene_circa_fdr.iloc[:,m])
						
						circa_tissues = []
						non_circa_tissues = []
						if path_gene_circa_count > 0:
							gene_in_circa = True
							# get path importance score
							total_path_count = total_path_count + 1
							path_score = path_search['paths'][k]['score']
							total_edge_score = total_edge_score + path_score
							# check whether the gene is circadian
							for n in range(0, tissue_len):
								if path_gene_circa_amp[n] >= amp_threshold and path_gene_circa_fdr[n] < fdr_threshold:
									circa_edge_score[n] = circa_edge_score[n] + path_score
									circa_tissues.append(query_tissue[n])		
								else:
									non_circa_tissues.append(query_tissue[n])
						else:
							gene_in_circa = False
							circa_tissues = non_circa_tissues = ['NaN']
					

						# fill in score details 
						node_ids = path_search['paths'][k]['node_ids']
						node_names = []
						for ni in node_ids:
							ni_name = str(path_search['nodes'][str(ni)]['properties']['name'])
							node_names.append(ni_name)
						rel_ids = path_search['paths'][k]['rel_ids']
						rel_types = []
						for ri in rel_ids:
							ri_name = str(path_search['relationships'][str(ri)]['rel_type'])
							rel_types.append(ri_name)
						path_details = {'source_node': node_names[0],
								'target_node': node_names[-1],
								'metapath': gene_metapath[j],
								'node_ids': ','.join([str(x) for x in node_ids]),
								'node_names': ','.join(node_names),
								'rel_ids': ','.join([str(x) for x in rel_ids]),
								'rel_names': ','.join(rel_types),
								'gene_symbol': ','.join(str(x) for x in path_gene_name),
								'whether_in_circadb': gene_in_circa,
								'circadian_tissue': ','.join(circa_tissues),
								'non_circadian_tissue': ','.join(non_circa_tissues),
								'importance_score': path_search['paths'][k]['score']
                                                                }
						score_details.append(path_details)
					
			# calculate proportion of circadian paths
			output = null_output
			output['total_meta_count'] = total_meta_count
			output['total_path_count'] = total_path_count
			if total_meta_count == 0:
				output['note'] = 'query drug and disease not connected by genes in hetionet'
			else:
				if total_path_count == 0:
					output['note'] = 'query drug and disease connected by genes not in CircaDB' 		
				elif total_path_count < min_path_count:
					output['note'] = 'query drug and disease connected by too few paths'
				else:
					edge_circa_score = circa_edge_score/total_edge_score
					output['edge_circa_score'] = dict((query_tissue[x], edge_circa_score[x]) for x in range(0, len(query_tissue)))
					detail_df = pd.DataFrame(score_details)
					detail_cols = ['source_node','target_node','metapath','node_ids','node_names','rel_ids','rel_names','gene_symbol','whether_in_circadb','circadian_tissue','non_circadian_tissue','importance_score']
					output['score_details'] = detail_df[detail_cols]	
	return output
