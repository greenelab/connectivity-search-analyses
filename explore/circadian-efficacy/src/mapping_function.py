# !/usr/bin/env python
# created by Yun Hao @GreeneLab 2019

import numpy as np
import pandas as pd

# map treatments to drugbank ID 
def map_drugbank_id(query_list, map_db_id, map_drug):
    # query_list: query treatments list; map_db_id: drugbank ID list; map_drug: drug name list

    for i in range(0,len(map_drug)):
        map_drug[i] = map_drug[i].lower()
    
    query_ids = []
    for i in range(0,len(query_list)):
        query_drugs = query_list[i].split(', ')
        drug_ids = []
        for j in range(0,len(query_drugs)):
            if query_drugs[j] in map_drug:
                drug_id = map_drug.index(query_drugs[j])
                drug_ids.append(map_db_id[drug_id])
            else:
                drug_ids.append('NA')
        drug_id_string = ', '.join(di for di in drug_ids)
        query_ids.append(drug_id_string)

    return query_ids

# map therapeutic area to disease ontology ID
def map_disease_id(query_list, map_do_id, map_disease):
    # query_list: query area list; map_do_id: DO ID list; map_disease: disease name

    query_ids = []
    for i in range(0,len(query_list)):
        query_id = map_disease.index(query_list[i])
        query_ids.append(map_do_id[query_id])

    return query_ids

# 
def map_gtex_expression(query_gene_list, query_tissue_list, map_gtex_tissue, map_tissue_name, gtex_exp):

    gtex_gene_name = list(gtex_exp.loc[:,'gene_id'])
    for i in range(0, len(gtex_gene_name)):
            gtex_gene_name[i] = gtex_gene_name[i].split('.')[0]

    query_gene_id_list = []
    for i in range(0, len(query_gene_list)):
        query_gene_id = gtex_gene_name.index(query_gene_list[i])
        query_gene_id_list.append(query_gene_id)

    exp_array = []
    for i in range(0, len(query_tissue_list)):
        query_tissue_id = map_tissue_name.index(query_tissue_list[i])
        query_tissues = map_gtex_tissue[query_tissue_id].split(', ')

        tissue_exp = gtex_exp.loc[:,query_tissues]
        gene_tissue_exp = tissue_exp.iloc[query_gene_id_list,:]
        gene_exp = gene_tissue_exp.mean(axis = 1)
        exp_array.append(list(gene_exp))

    return exp_array

