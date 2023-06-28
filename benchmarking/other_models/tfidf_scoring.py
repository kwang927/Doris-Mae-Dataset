from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
import re
import numpy as np
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm

def preprocessing(text):
    '''
    Note, this preprocessing function should be applied to any text before getting its embedding. 
    Input: text as string
    
    Output: removing newline, latex $$, and common website urls. 
    '''
    pattern = r"(((https?|ftp)://)|(www.))[^\s/$.?#].[^\s]*"
    text = re.sub(pattern, '', text)
    return text.replace("\n", " ").replace("$","")

def all_ranking_tfidf(vectorizer, q, abstract_l, q_candidates, query_mode):
    if query_mode == "paragraph":
        query_l = vectorizer.transform([q])
        similarities = cosine_similarity(query_l, abstract_l)[0]
    elif query_mode == "sentence":
        sent_text = sent_tokenize(q)
        query_sent = [s for s in sent_text if len(preprocessing(s)) >= 5]
        similarity_list = []
        for sentence in query_sent:
            query_l = vectorizer.transform([sentence])
            similarity = cosine_similarity(query_l, abstract_l)[0]
            similarity_list.append(similarity)
        similarities = np.mean(similarity_list, axis=0)
        
    sorted_indices = np.argsort(similarities)[::-1]
    ranked_list = [q_candidates[idx] for idx in sorted_indices]
    return ranked_list

def all_ranking_result_tfidf(dataset, query_mode):
    query_lists = [q['query_text'] for q in dataset['Query']]
    candidate_list = [q['candidate_pool'] for q in dataset['Query']]
    vectorizer = TfidfVectorizer()
    rank_dict_list = []
    abstracts = dataset['Corpus']
    for i in tqdm(range(len(query_lists))):
        rank_dict = {}
        q = query_lists[i]
        q_candidates = candidate_list[i]
        abs_list = [preprocessing(abstracts[idx]['original_abstract']) for idx in q_candidates]
        abstract_tfidf = vectorizer.fit_transform(abs_list)
        ranked_list = all_ranking_tfidf(vectorizer, q, abstract_tfidf, q_candidates, query_mode)
        rank_dict['index_rank'] = ranked_list
        rank_dict_list.append(rank_dict)
        
    return rank_dict_list
