import json
import pickle
import numpy as np
import rocketqa
import re
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize

def preprocessing(text):
    '''
    Note, this preprocessing function should be applied to any text before getting its embedding. 
    Input: text as string
    
    Output: removing newline, latex $$, and common website urls. 
    '''
    pattern = r"(((https?|ftp)://)|(www.))[^\s/$.?#].[^\s]*"
    text = re.sub(pattern, '', text)
    return text.replace("\n", " ").replace("$","")

def all_ranking_RQA(cross_encoder, q, abstract_l, q_candidates, query_mode):
    '''
    Input: 
        cross_encoder: loaded rocketqa encoder model
        q: query text
        abstract_l: list of abstract text for given query
        q_candidates: list of abstracts ids for given query
        query_mode: text embedding mode for query text
    
    Return:
        returning ranked_list, ranked_list is a sorted list of the ids corresponds to abstracts
    '''
    if query_mode == 'paragraph':
        query_l = [q] * len(q_candidates)
        relevance = cross_encoder.matching(query=query_l, para=abstract_l)
        similarities = list(relevance)
    elif query_mode == 'sentence':
        sent_text = sent_tokenize(q)
        query_sent = [s for s in sent_text if len(preprocessing(s)) >= 5]
        similarity_list = []
        for sentence in query_sent:
            query_l = [sentence] * len(q_candidates)
            relevance = cross_encoder.matching(query=query_l, para=abstract_l)
            similarity = list(relevance)
            similarity_list.append(similarity)
        similarities = np.mean(similarity_list, axis=0)
    
    sorted_indices = np.argsort(similarities)[::-1]
    ranked_list = [q_candidates[idx] for idx in sorted_indices]
    return ranked_list


def all_ranking_result_RQA(dataset, query_mode):
    '''
    Input: 
        dataset: doris mae dataset
        query_mode: text embedding mode for query text
    
    Return:
          returning rank_dict_list, rank_dict_list is a list of dictionary, the position index corresponds to the number of query, and in each dicitonary, the only key is index_rank, the value is a list of paper abstarct ids sorted in descending order w.r.t. their relevance of the given query, based on the rocketqa model's evaluation. 
    '''
    cross_encoder = rocketqa.load_model(model="v2_marco_ce", use_cuda=False, device_id=0, batch_size=16)
    query_lists = [q['query_text'] for q in dataset['Query']]
    candidate_list = [q['candidate_pool'] for q in dataset['Query']]
    abstracts = dataset['Corpus']
    rank_dict_list = []
    for i in tqdm(range(len(query_lists))):
        rank_dict = {}
        q = query_lists[i]
        q_candidates = candidate_list[i]
        abs_list = [preprocessing(abstracts[idx]['original_abstract']) for idx in q_candidates]
        ranked_list = all_ranking_RQA(cross_encoder, q, abs_list, q_candidates, query_mode)
        rank_dict['index_rank'] = ranked_list
        rank_dict_list.append(rank_dict)
        
    return rank_dict_list
    