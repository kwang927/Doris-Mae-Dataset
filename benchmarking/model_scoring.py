import numpy as np
from tqdm import tqdm
import os
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pickle
import json
from itertools import chain
import multiprocessing as mp

from config import *
from model_embedding import *
from aspire.aspire_distance import find_ot_distance

from other_models.bm25_scoring import all_ranking_result_bm25
from other_models.tfidf_scoring import all_ranking_result_tfidf
# from other_models.rocketqa_scoring import all_ranking_result_RQA


rank_result_path = "dataset/rankings"
embedding_result_path = "dataset/embeddings"

def text2embedding(text, embedding_dict, mode = "paragraph"):
    '''
    Input: text as string to be embedded
           embedding_dict is a mapping between strings to embeddings
           mode: "sentence" or "paragraph"
           
    Return: list of vector embeddings for the given text, list length could be one or more
    '''
    if mode == "paragraph":
        ret = embedding_dict[text]
        if len(ret.shape)==2:
            ret = list(ret)
        elif len(ret.shape)==1:
            ret = [ret]
        else:
            raise ValueError("values in embedding dictionary has wrong shape, not 1 or 2")
    
    elif mode == "sentence":
        text_list = sent_tokenize(text)
        ret = []
        for t in text_list:
            if t in embedding_dict.keys():
                t_embedding = embedding_dict[t]
                assert len(t_embedding.shape)==1, "embedding shape mismatch"
                ret.append(t_embedding)
    return ret


def get_relevance_score(query_vec, abstract_vec, metric = "cosine", aggregation = "max_max"):
    '''
    Input: query_vec is a list of embedding vectors for one query
           abstract_vec is a list of embedding vectors for one abstract
    
    Return: a scalar value that is the similarity between query vec and abstract vec.
    '''
    if metric == "ot":
        return find_ot_distance(np.array(query_vec), np.array(abstract_vec))

    if metric== "cosine":
        ret = cosine_similarity(np.array(query_vec), np.array(abstract_vec))
    elif metric == "l2":
        ret = -1*euclidean_distances(np.array(query_vec), np.array(abstract_vec))

            
    if aggregation == "max_max":
        return np.max(ret)
    elif aggregation == "mean_max":
        return np.mean(np.max(ret, axis= 1))
    else:
        raise ValueError("undefined aggregation")
        

def get_relevance_score_for_all(model_name, query_list, query_id_list, candidate_pool_list, candidate_pool_id_list, embedding_dict, config):
    '''
    Input: 
          model_name: the name of the model
          query_list: list of queries (or subqueries)
          query_id_list: list of query ids, corresponding to queries
          candidate_pool_list: list of candidate pool, where each candidate pool is a list of abstracts in string
          candidate_pool_id_list: list of candidate pool id, where each is a list of ids, each id is the paper id in dataset.
          embedding_dict: calculated from model_embedding.py, key is text, value is numpy ndarray. 
          config: the configuration dict
    Return:
          get a list of dictionary, the position index corresponds to the query_id list, where each dictionary is a mapping between (key= abstract id), value(= relevance score conditioned on the given query)/
    '''
    metric = config['model_name_dict'][model_name]['metric']
    aggregation = config['model_name_dict'][model_name]['aggregation']
    query_mode = config['model_name_dict'][model_name]['query_mode']
    abstract_mode = config['model_name_dict'][model_name]['abstract_mode']
    
    ret=[]
    for i in tqdm(range(len(query_list)),desc="calculating relevance score...", leave = False):
        query, query_id, candidate_pool, abstract_id_list = query_list[i], query_id_list[i], candidate_pool_list[i], candidate_pool_id_list[i]
        tmp_ret = {}
        query_vec = text2embedding(query, embedding_dict, query_mode)
        candidate_pool_vec_list = [text2embedding(abstract, embedding_dict, abstract_mode) for abstract in candidate_pool]
        for i in range(len(candidate_pool_vec_list)):
            rel_score = get_relevance_score(query_vec, candidate_pool_vec_list[i], metric, aggregation)
            tmp_ret[abstract_id_list[i]] = rel_score
        ret.append(tmp_ret)
            
    return ret

def get_text_list(dataset):
    '''
    Input: the doris mae dataset, note there are 4 abstract ids removed, because their tokens exceed 512, which is what bert variants allowed. # 205580, 346695, 346826, 346836 are paper ids that are more than 512 tokens, making it unfair for bert-variant models. 
    
    Return: 
          query_list: list of queries (or subqueries)
          query_id_list: list of query ids, corresponding to queries
          candidate_pool_list: list of candidate pool, where each candidate pool is a list of abstracts in string
          candidate_pool_id_list: list of candidate pool id, where each is a list of ids, each id is the paper id in dataset.
    '''
    removed_list = [205580, 346695, 346826, 346836, 28674]
    
    Query = dataset['Query']
    query_list, query_id_list, candidate_pool_list, candidate_pool_id_list =[], [], [], []
    for i in range(len(Query)):
        query_list.append(Query[i]['query_text'])
        query_id_list.append(i)
        candidate_pool_id_list.append([idx for idx in Query[i]['candidate_pool'] if idx not in removed_list]) 
        candidate_pool_list.append([dataset['Corpus'][idx]['original_abstract'] for idx in Query[i]['candidate_pool'] if idx not in removed_list])
    return query_list, query_id_list, candidate_pool_list, candidate_pool_id_list
        

def rank_by_model(dataset, model_name, config):
    '''
    Input: 
           dataset: doris mae dataset
           model_name: model name to be evaluated
           config: configuration dictionary
    Return:
          returning rank, rank is a list of dictionary, the position index corresponds to the number of query, and in each dicitonary, the only key is index_rank, the value is a list of paper abstarct ids sorted in descending order w.r.t. their relevance of the given query, based on the current model's evaluation. 
    '''
    query_mode = config["model_name_dict"][model_name]["query_mode"]
    abstract_mode = config["model_name_dict"][model_name]["abstract_mode"]
    aggregation = config["model_name_dict"][model_name]["aggregation"]
    level = config["level"]
    if not os.path.exists(f"{rank_result_path}/{level}"):
        os.makedirs(f"{rank_result_path}/{level}")
    
    if not os.path.exists(f"{embedding_result_path}/{level}"):
        os.makedirs(f"{embedding_result_path}/{level}")
    
    if os.path.exists(f"{rank_result_path}/{level}/ranking_{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation}.pickle"):
        with open(f"{rank_result_path}/{level}/ranking_{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation}.pickle", "rb") as f:
            rank = pickle.load(f)
        print(f"ranking results for {model_name} are already calculated")
        return rank
    
    elif model_name in ["rocketqa", "tfidf", "bm25", "GPT4", "GPT3.5", "deanno"]:
        if model_name == "rocketqa":
            rank = all_ranking_result_RQA(dataset, abstract_mode, query_mode)
        elif model_name == "tfidf":
            rank = all_ranking_result_tfidf(dataset, query_mode)
        elif model_name == "bm25":
            rank = all_ranking_result_bm25(dataset, query_mode)
            
        with open(f"{rank_result_path}/{level}/ranking_{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation}.pickle", "wb") as f:
            pickle.dump(rank, f)
        return rank

    
    elif model_name in ["specter", "ada", "llama", "specter-ID"]:
        with open(f"{embedding_result_path}/{level}/embedding_{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation}.pickle", "rb") as f:
            embedding_dict = pickle.load(f)
            print(f"embedding for {model_name} is already calculated")
        query_list, query_id_list, candidate_pool_list, candidate_pool_id_list = get_text_list(dataset)
            
    elif model_name in config['model_name_dict'].keys(): 
        
        query_list, query_id_list, candidate_pool_list, candidate_pool_id_list = get_text_list(dataset)
        
        if os.path.exists(f"{embedding_result_path}/{level}/embedding_{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation}.pickle"):
            with open(f"{embedding_result_path}/{level}/embedding_{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation}.pickle", "rb") as f:
                embedding_dict = pickle.load(f)
                print(f"embedding for {model_name} is already calculated")
        else:
              
            abstract_list = list(set(list(chain.from_iterable(candidate_pool_list))))
            
            pre_encoded_abstract = pre_encoding_by_mode(abstract_list, mode = config['model_name_dict'][model_name]['abstract_mode'])
            pre_encoded_query = pre_encoding_by_mode(query_list, mode = config['model_name_dict'][model_name]['query_mode'])
                

            pre_encoded_text = pre_encoded_abstract + pre_encoded_query
            
            print(f"loading model: {model_name}....")
            model, tokenizer = load_model(model_name, config['cuda'])
            embedding_dict = get_embedding(model_name, model, tokenizer, pre_encoded_text, config['cuda'], config['bs'])
            
            with open(f"{embedding_result_path}/{level}/embedding_{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation}.pickle", "wb") as f:
                pickle.dump(embedding_dict, f)
    
    score_dict_list= get_relevance_score_for_all(model_name, query_list, query_id_list, candidate_pool_list, candidate_pool_id_list, embedding_dict, config)
    rank =[]
    scores_rank = []
    for score_dict in score_dict_list:
        sorted_list = sorted(score_dict.items(), key = lambda x: x[1], reverse = True)
        tmp_rank = {"index_rank": [s[0] for s in sorted_list]}
        rank.append(tmp_rank)
        scores_rank.append(sorted_list)
    
    with open(f"{rank_result_path}/{level}/ranking_{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation}.pickle", "wb") as f:
        pickle.dump(rank, f)   
    return rank
   
