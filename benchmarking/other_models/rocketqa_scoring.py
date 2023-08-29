import json
import pickle
import numpy as np
import rocketqa
import re
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize
from itertools import chain


def preprocessing(text):
    '''
    Note, this preprocessing function should be applied to any text before getting its embedding. 
    Input: text as string
    
    Output: removing newline, latex $$, and common website urls. 
    '''
    pattern = r"(((https?|ftp)://)|(www.))[^\s/$.?#].[^\s]*"
    text = re.sub(pattern, '', text)
    return text.replace("\n", " ").replace("$","")

def pre_encoding_by_mode(text, mode = "paragraph"):
    '''
    Input: text as a list of strings, waiting to be embedded
           model: default is processing by paragraph, for models such as sentbert or ance, or by user preference
                  another mode is "sentence", where each string is splits into sentences, and return a flattened
                  list of sentences, with too short sentences are eliminated. 
    Return: list of strings, where strings of less than 5 chars are removed if a model is processing it sentence by sentence.
    '''
    if mode == "paragraph":
        return text
    elif mode == "sentence":
        sent_text = list(chain.from_iterable([sent_tokenize(t) for t in text]))
        ret = [s for s in sent_text if len(preprocessing(s))>=5]
        return ret
    else:
        raise ValueError("only supports paragraph or sentence processing modes")

def all_ranking_RQA(cross_encoder, q, abstract_l, q_candidates, query_mode):
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


def all_ranking_result_RQA(dataset, abstract_mode, query_mode):
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
        asb_list = pre_encoding_by_mode(abs_list, abstract_mode)
        ranked_list = all_ranking_RQA(cross_encoder, q, abs_list, q_candidates, query_mode)
        rank_dict['index_rank'] = ranked_list
        rank_dict_list.append(rank_dict)
        
    return rank_dict_list
    
