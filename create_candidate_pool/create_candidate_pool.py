import pickle
from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import string
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from rank_bm25 import BM25Okapi
import pdb
import argparse


def ensure_directory_exists(dir_path):
    """
    Ensure that the directory specified by dir_path exists.
    If it doesn't, create it.
    
    Parameters:
    - dir_path (str): The path of the directory to check/create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory {dir_path} created.")
    else:
        print(f"Directory {dir_path} already exists.")


def clean_sentence(text):
    """
    Clean the given text by converting it to lowercase and replacing punctuations with spaces.
    
    Parameters:
    - text (str): The text to be cleaned.

    Returns:
    - str: The cleaned text.
    """
    text = text.strip().lower()
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    return text


def get_all_vocab(abstracts):
    """
    Extract the vocabulary from a list of abstracts.
    
    Parameters:
    - abstracts (list of str): The list of abstracts.

    Returns:
    - set: The set of unique words across all abstracts.
    """
    vocab = []
    for text in abstracts:
        text = clean_sentence(text)
        tmp = word_tokenize(text)
        vocab += tmp
    return set(vocab)


def filter_words(vocab):
    """
    Filters the given vocabulary by excluding words containing two or more numbers.
    
    Parameters:
    - vocab (set): The set of words to be filtered.

    Returns:
    - dict: A dictionary mapping from words to their corresponding indices.
    """
    ret = []
    for w in list(vocab):
        contains_two_number = False
        count_number = 0
        for char in list(w):
            if char in list("0123456789"):
                count_number += 1
            if count_number >= 2:
                contains_two_number = True
                break
        if not contains_two_number:
            ret.append(w)
    ret = list(set(ret))
    word2idx = {}
    for i, w in enumerate(ret):
        word2idx[w] = i
    return word2idx


def get_word2idf(data, word2idx):
    """
    Calculate the inverse document frequency (IDF) of each word in the vocabulary.
    
    Parameters:
    data (list of str): The list of texts.
    word2idx (dict): A dictionary mapping words to their indices.

    Returns:
    dict: A dictionary mapping words to their IDF values.
    """
    ret = {}
    for w in word2idx:
        ret[w] = 0
    for text in data:
        text = clean_sentence(text)
        tmp = []
        for t in word_tokenize(text):
            if t in ret.keys() and t not in tmp:
                ret[t] += 1
                tmp.append(t)
    for w in ret.keys():
        if ret[w] != 0:
            ret[w] = np.log(len(data)) - np.log(ret[w])
    return ret


# Compute the tfidf score for a single token
def get_tfidf_weight(token_counts, token, length, word2idf):
    """
    Compute the TF-IDF score for a given token.
    
    Parameters:
    - token_counts (dict): A dictionary mapping tokens to their counts.
    - token (str): The token for which the score is to be calculated.
    - length (int): The total number of tokens.
    - word2idf (dict): A dictionary mapping words to their IDF values.

    Returns:
    - float: The TF-IDF score of the given token.
    """
    if token in token_counts.keys() and token in word2idf.keys():
        tf = token_counts[token] / length
        idf = word2idf[token]
        return tf * idf
    else:
        return 0


# Tokenize all abstracts
def tokenize_abstracts(abstract):
    """
    Tokenize a list of abstracts.
    
    Parameters:
    - abstract (list of str): The list of abstracts to be tokenized.

    Returns:
    - list of list of str: The tokenized abstracts.
    """
    tokenize_abstract = []
    for doc in tqdm(abstract, desc="Tokenizing corpus"):
        doc = clean_sentence(doc)
        tokenize_abstract.append(word_tokenize(doc))
    return tokenize_abstract


# Tokenize query
def tokenize_query(query):
    """
    Tokenize the given query.
    
    Parameters:
    - query (str): The query to be tokenized.

    Returns:
    - list of str: The tokenized query.
    """
    query = clean_sentence(query)
    return word_tokenize(query)





# ================ TFIDF query-level ================

def fabs_tfidf(abstracts):
    """
    Calculate the TF-IDF vectors for a list of abstracts.
    
    Parameters:
    - abstracts (list of str): The list of abstracts.

    Returns:
    - tuple: The TF-IDF vectors of abstracts and the vectorizer.
    """
    # Calculate TF-IDF
    vectorizer = TfidfVectorizer()
    abstract_tfidf = vectorizer.fit_transform(abstracts)
    #     query_tfidf = vectorizer.transform(query_list)

    return abstract_tfidf, vectorizer


def ranking(query_emb, abstract_emb, k):
    """
    Rank the abstracts based on cosine similarity to the query embedding.
    
    Parameters:
    - query_emb (array-like): The embedding of the query.
    - abstract_emb (array-like): The embeddings of the abstracts.
    - k (int): The number of top abstracts to retrieve.

    Returns:
    - list of int: The indices of the top-k abstracts.
    """

    # Calculate the cosine similarity
    # pdb.set_trace()
    similarities = cosine_similarity(query_emb.reshape(1, -1), abstract_emb)[0]

    # Sort the abstracts
    sorted_indices = np.argsort(similarities)[::-1]
    top_k_indices = sorted_indices[:k]
    return top_k_indices




# ================ BM25 query-level ================
def find_index(str_list, abs_list):
    """
    Find the indices of strings in str_list within abs_list.
    
    Parameters:
    - str_list (list of str): List of strings whose indices are to be found.
    - abs_list (list of str): List in which to find the indices.

    Returns:
    - list of int: The indices of str_list in abs_list.
    """
    result = []
    for s in str_list:
        result.append(abs_list.index(s))
    return result


def top_k_ind_bm25(query, abstracts, k, BM25_abs):
    """
    Get the top-k abstract indices based on BM25 ranking.
    
    Parameters:
    - query (str): The query for which to find relevant abstracts.
    - abstracts (list of str): The list of abstracts.
    - k (int): The number of abstracts to retrieve.
    - BM25_abs (object): The BM25 object.

    Returns:
    - list of int: The indices of the top-k abstracts.
    """
    tokenized_query = word_tokenize(query)
    str_result = BM25_abs.get_top_n(tokenized_query, abstracts, n=k)
    return find_index(str_result, abstracts)





# ================ functions for aspect-based candidate pool ================

def find_element_index(lst, elem):
    """
    Find the index of an element in a list. If not found, return -1.
    
    Parameters:
    - lst (list): The list in which to find the element.
    - elem (any): The element to find.

    Returns:
    - int: The index of the element, or -1 if not found.
    """

    """Returns the index of the given element in the list, or -1 if the element is not in the list"""
    try:
        return lst.index(elem)
    except ValueError:
        return -1


def find_mean_reciprocal_rank(ind, result_list):
    """
    Compute the mean reciprocal rank (MRR) based on the provided result list.

    Parameters:
    - ind (int): The index to be checked within the result list.
    - result_list (list): List of results containing various indices.

    Returns:
    - tuple: Contains two integers, the count of valid positions and the computed MRR.
    """
    result = 0
    count = 0
    for res in result_list:
        #         print(type(res))
        position = find_element_index(list(res), ind)

        if position != -1:
            count += 1
            result += 1 / (position + 1)

    return count, result


def create_temp_dataset(dataset):
    """
    Create a temporary dataset from the provided dataset.

    Parameters:
    - dataset (dict): A dictionary containing various queries.

    Returns:
    - list: A list of dictionaries representing the new dataset format.
    """
    ret = []
    for i in range(len(dataset['Query'])):
        temp_dict = {}
        temp_dict['ID'] = dataset['Query'][i]['idea_from']
        temp_dict['Query'] = dataset['Query'][i]['query_text']
        temp_dict['Reqs_in_id'] = dataset['Query'][i]['aspects']
        temp_dict['ada_embedding'] = dataset['Query'][i]['ada_embedding']

        ret.append(temp_dict)

    return ret






# ================ TF-IDF aspect-level ================



# ================ BM25 aspect-level ================




# ================ functions to filter out candidates with irrelevant categories  ================
def in_category(query_id, abstract_id, real_dataset, corpus):
    """
    Check if an abstract belongs to the same category as the query.

    Parameters:
    - query_id (int): ID of the query.
    - abstract_id (int): ID of the abstract.
    - real_dataset (list): List of actual data entries.
    - corpus (list): List of all available abstracts.

    Returns:
    - bool: True if the abstract belongs to the same category as the query, else False.
    """
    g_id = real_dataset[query_id]["ID"]
    g_cat = []
    for i in g_id:
        g = corpus[i]
        g_cat += g['categories']
    abs_cat = corpus[abstract_id]['categories']
    if len(set(g_cat).intersection(set(abs_cat))) > 0:
        return True
    else:
        return False


def get_filtered(mode, ranks, real_dataset, corpus):
    """
    Filter out candidates based on their relevance to the query category.

    Parameters:
    - mode (str): The mode of filtering ("query", "aspect", or "sorted").
    - ranks (list): List of ranked abstracts.
    - real_dataset (list): List of actual data entries.
    - corpus (list): List of all available abstracts.

    Returns:
    - list: List of filtered candidates.
    """
    dataset_length = len(real_dataset)
    count = 0
    total_count = 0
    filtered = []
    if mode == "query":
        for query_id in tqdm(range(dataset_length)):
            query_pool = ranks[query_id]
            new_query_pool = []
            for abs_id in query_pool:
                total_count += 1
                if in_category(query_id, abs_id, real_dataset, corpus):
                    new_query_pool.append(abs_id)
                else:
                    count += 1
            filtered.append(new_query_pool)
        #         print(count)
        #         print(total_count)
        #         print(len(filtered))
        return filtered
    elif mode == "aspect":
        for query_id in tqdm(range(dataset_length)):
            query_pool = ranks[query_id]
            new_query_pool = []
            for req_pool in query_pool:
                new_req_pool = []
                for abs_id in req_pool:
                    total_count += 1
                    if in_category(query_id, abs_id, real_dataset, corpus):
                        new_req_pool.append(abs_id)
                    else:
                        count += 1
                new_query_pool.append(new_req_pool)
            filtered.append(new_query_pool)
        #         print(count)
        #         print(total_count)
        return filtered

    elif mode == "sorted":
        for query_id in tqdm(range(dataset_length)):
            new_dict = {}
            for k in ranks[query_id].keys():
                new_list = []
                for abs_id in ranks[query_id][k]:
                    total_count += 1
                    if in_category(query_id, abs_id, real_dataset, corpus):
                        new_list.append(abs_id)
                    else:
                        count += 1

                new_dict[k] = new_list
            filtered.append(new_dict)

        #         print(total_count)
        #         print(count)
        return filtered


def get_top_k_from_sorted_req(sorted_req_dict, k):
    """
    Retrieve the top-k abstracts from the sorted requirements dictionary.

    Parameters:
    - sorted_req_dict (dict): Dictionary of sorted requirements.
    - k (int): Number of top abstracts to retrieve.

    Returns:
    - list: Top-k abstracts from the sorted requirements.
    """
    result = []
    max_count = max(sorted_req_dict.keys())

    for c in range(max_count, 0, -1):

        if c in sorted_req_dict.keys():

            for item in sorted_req_dict[c]:
                result.append(item)
                if (len(result) == k):
                    return result

    return result


def get_k_from_each_req(req_separated_list, k):
    """
    Retrieve k abstracts from each requirement in the separated list.

    Parameters:
    - req_separated_list (list): List of separated requirements.
    - k (int): Number of abstracts to retrieve from each requirement.

    Returns:
    - list: List of abstracts retrieved from each requirement.
    """
    result = []
    for i in req_separated_list:
        result += list(i[:k])

    return list(set(result))


def create_candidate_per_query(i, A, B, C, D, E, filtered_query_tfidf, filtered_req_tfidf_separated,
                               filtered_req_tfidf_sorted, filtered_query_bm25, filtered_req_bm25_separated,
                               filtered_req_bm25_sorted):
    """
    Create a candidate dataset for a specific query based on various retrieval techniques.

    Parameters:
    - i (int): Index of the query.
    - A, B, C, D, E (int): Various configuration parameters.
    - filtered_query_tfidf, filtered_req_tfidf_separated, etc. (various types): Various data structures to support the retrieval process.

    Returns:
    - tuple: Contains a dictionary representing the new candidate and the number of requirements.
    """
    temp_dict = {}

    req_num = len(filtered_req_tfidf_separated[i])
    #     print(req_num)
    temp_dict['TFIDF_query'] = list(filtered_query_tfidf[i][:A])
    temp_dict['BM25_query'] = list(filtered_query_bm25[i][:A])

    temp_dict['TFIDF_req_score'] = get_top_k_from_sorted_req(filtered_req_tfidf_sorted[i], B)
    temp_dict['BM25_req_score'] = get_top_k_from_sorted_req(filtered_req_bm25_sorted[i], B)

    temp_dict['TFIDF_req_separate'] = get_k_from_each_req(filtered_req_tfidf_separated[i], C)
    temp_dict['BM25_req_separate'] = get_k_from_each_req(filtered_req_bm25_separated[i], C)

    tmp = []
    tmp += temp_dict['TFIDF_query']
    tmp += temp_dict['BM25_query']
    tmp += temp_dict['TFIDF_req_score']
    tmp += temp_dict['BM25_req_score']
    tmp += temp_dict['TFIDF_req_separate']
    tmp += temp_dict['BM25_req_separate']

    whole_list = list(filtered_query_tfidf[i]) + list(filtered_query_bm25[i]) \
                 + get_top_k_from_sorted_req(filtered_req_tfidf_sorted[i], 1000000) \
                 + get_top_k_from_sorted_req(filtered_req_bm25_sorted[i], 1000000) \
                 + get_k_from_each_req(filtered_req_tfidf_separated[i], 500) \
                 + get_k_from_each_req(filtered_req_bm25_separated[i], 500)

    whole_set = set(whole_list)
    temp_dict['TFIDF_and_BM25'] = list(set(tmp))
    #     print(whole_set.difference(set(tmp)))
    temp_dict['unused'] = list(whole_set.difference(set(tmp)))

    return temp_dict, req_num


# ================ Construct Candidate Pool ================


# ================ Citation Signals ================



# ================ 20 candidates from Ada ================



def cosine_similarity_ada(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.
    
    Parameters:
    - vec1 (array-like): The first vector.
    - vec2 (array-like): The second vector.

    Returns:
    - float: The cosine similarity between the two vectors.
    """

    
    """
    Calculate the cosine similarity between two vectors using numpy
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)

    if not magnitude_vec1 or not magnitude_vec2:
        # In case one of the vectors has 0 magnitude, return 0
        return 0
    return dot_product / (magnitude_vec1 * magnitude_vec2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output_dir', required=True, help='The directory to output directory')
    parser.add_argument('-q', '--dataset', required=True, help='The dataset to read in')
    parser.add_argument('-a','--ada_emb',required=True, help='The the ada_embedding to read in')


    args = parser.parse_args()

    output_path = args.output_dir
    dataset_path = args.dataset
    ada_emb_path = args.ada_emb

    # ======================Load data======================

    ensure_directory_exists(output_path)

    # output_path = "test_create_cand_pool"

    # Change the path to the dataset if needed
    print("Loading data...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    print("Data loaded")
    
    dataset_length = len(dataset['Query'])

    with open(ada_emb_path, 'rb') as f:
        ada_emb = pickle.load(f)

    for k in ada_emb:
        if k == 'Query':
            for q_id in ada_emb[k].keys():
                dataset['Query'][q_id]['ada_embedding'] = ada_emb[k][q_id]
        elif k == 'Corpus':
            for a_id in ada_emb[k].keys():
                dataset['Corpus'][a_id]['ada_embedding'] = ada_emb[k][a_id]


    corpus = dataset['Corpus']

    if os.path.exists(f"./{output_path}/tokenized_abstract.pickle"):
        print(f"The file exists!")
        with open(f"./{output_path}/tokenized_abstract.pickle", 'rb') as f:
            all_abs = pickle.load(f)
    else:
        print(f"The file does not exist!")
        abstracts = []
        for each in corpus:
            abstracts.append(each['original_abstract'])
        all_abs = tokenize_abstracts(abstracts)
        # TODO: Change the path later
        with open(f"./{output_path}/tokenized_abstract.pickle", 'wb') as f:
            pickle.dump(all_abs, f)

    cleaned_abs = [" ".join(each) for each in all_abs]
    queries = [q['query_text'].lower().strip() for q in dataset['Query']]



    # ======================TF-IDF query level======================

    if os.path.exists(f"./{output_path}/query_level_candidate_tfidf.pickle"):
        print("query_level_candidate_tfidf.pickle exists")
        with open(f"./{output_path}/query_level_candidate_tfidf.pickle", 'rb') as f:
            query_ranking_tfidf = pickle.load(f)
    else:

        print("Initializing TF-IDF...")
        abs_tfidf, vectorizer = fabs_tfidf(cleaned_abs)
        print("TF-IDF initialized")
        query_tfidf = vectorizer.transform(queries)

        query_ranking_tfidf = []
        for i in tqdm(range(dataset_length), desc="Calculating TF-IDF query-level candidates"):
            query_ranking_tfidf.append(ranking(query_tfidf[i], abs_tfidf, 50))

        # TODO: change the path
        with open(f"./{output_path}/query_level_candidate_tfidf.pickle", 'wb') as f:
            pickle.dump(query_ranking_tfidf, f)

    # ======================BM25 query level======================
    if os.path.exists(f"./{output_path}/query_level_candidate_bm25.pickle"):
        print("query_level_candidate_bm25.pickle exists")
        with open(f"./{output_path}/query_level_candidate_bm25.pickle", 'rb') as f:
            query_ranking_bm25 = pickle.load(f)

    else:
        print("Initializing BM25 ...")
        bm25_abs = BM25Okapi(all_abs)
        print("BM25 initialized")

        query_ranking_bm25 = []
        for i in tqdm(range(dataset_length), desc="Calculating BM25 query-level candidates"):
            query_ranking_bm25.append(top_k_ind_bm25(queries[i], cleaned_abs, 50, bm25_abs))

        # TODO: change the path
        with open(f"./{output_path}/query_level_candidate_bm25.pickle", 'wb') as f:
            pickle.dump(query_ranking_bm25, f)


    # ======================TF-IDF aspect level======================

    temp_dataset = create_temp_dataset(dataset)

    aspects = []
    for q in dataset['Query']:
        asp_idx = list(q['aspects'].keys())
        aspects.append([dataset['aspect_id2aspect'][idx] for idx in asp_idx])

    if os.path.exists(f"./{output_path}/aspect_separate_candidate_tfidf.pickle"):
        print("/aspect_separate_candidate_tfidf.pickle exists")
        with open(f"./{output_path}/aspect_separate_candidate_tfidf.pickle", 'rb') as f:
            tfidf_aspect_separate = pickle.load(f)

    else:

        tfidf_aspect_separate = []
        for i in tqdm(aspects, desc='Calculating TF-IDF aspect-level candidates'):
            aspect_tfidf = vectorizer.transform(i)
            one_query_rank = []
            for each in aspect_tfidf:
                one_aspect = ranking(each, abs_tfidf, 500)
                one_query_rank.append(one_aspect)
            tfidf_aspect_separate.append(one_query_rank)

        # TODO: change the path
        with open(f"./{output_path}/aspect_separate_candidate_tfidf.pickle", 'wb') as f:
            pickle.dump(tfidf_aspect_separate, f)

    if os.path.exists(f"./{output_path}/aspect_sorted_candidate_tfidf.pickle"):
        with open(f"./{output_path}/aspect_sorted_candidate_tfidf.pickle", 'rb') as f:
            tfidf_sorted_aspects = pickle.load(f)
    else:
        print("Initializing TF-IDF...")
        abs_tfidf, vectorizer = fabs_tfidf(cleaned_abs)
        print("TF-IDF initialized")

        reciprocal_list = []
        tfidf_sorted_aspects = []
        # corpus = dataset['Corpus']

        for aspect_list in tqdm(tfidf_aspect_separate):
            reciprocal_dict = {}
            count_dict = {}
            for ranks in aspect_list:
                for i, idx in enumerate(ranks):
                    if idx not in count_dict:
                        count_dict[idx] = 0
                    if idx not in reciprocal_dict:
                        reciprocal_dict[idx] = 0
                    count_dict[idx] += 1
                    reciprocal_dict[idx] += 1 / (i + 1)

            reversed_dict = {}
            for key, value in count_dict.items():
                if value not in reversed_dict:
                    reversed_dict[value] = [key]
                else:
                    reversed_dict[value].append(key)

            sorted_count_dict = {}
            for key in reversed_dict:
                sorted_value = sorted(reversed_dict[key], key=lambda x: (reciprocal_dict[x], -x), reverse=True)
                sorted_count_dict[key] = sorted_value

            reciprocal_list.append(reciprocal_dict)
            tfidf_sorted_aspects.append(sorted_count_dict)

        # TODO: change the path
        with open(f"./{output_path}/aspect_sorted_candidate_tfidf.pickle", 'wb') as f:
            pickle.dump(tfidf_sorted_aspects, f)


    # ======================BM25 aspect level======================

    if os.path.exists(f"./{output_path}/aspect_separate_candidate_bm25.pickle"):
        print("/aspect_separate_candidate_bm25.pickle exists")
        with open(f"./{output_path}/aspect_separate_candidate_bm25.pickle", 'rb') as f:
            bm25_aspect_separate = pickle.load(f)

    else:
        print("Initializing BM25 ...")
        bm25_abs = BM25Okapi(all_abs)
        print("BM25 initialized")

        bm25_aspect_separate = []
        for i in tqdm(aspects, desc='Calculating BM25 aspect-level candidates'):
            one_query_rank = []
            for each in i:
                one_aspect = top_k_ind_bm25(each, cleaned_abs, 500, bm25_abs)
                one_query_rank.append(one_aspect)
            bm25_aspect_separate.append(one_query_rank)

        # TODO: change the path
        with open(f"./{output_path}/aspect_separate_candidate_bm25.pickle", 'wb') as f:
            pickle.dump(bm25_aspect_separate, f)

    if os.path.exists(f"./{output_path}/aspect_sorted_candidate_bm25.pickle"):
        with open(f"./{output_path}/aspect_sorted_candidate_bm25.pickle", 'rb') as f:
            bm25_sorted_aspects = pickle.load(f)
    else:

        reciprocal_list = []
        bm25_sorted_aspects = []
        # corpus = dataset['Corpus']

        for aspect_list in tqdm(bm25_aspect_separate):
            reciprocal_dict = {}
            count_dict = {}
            for ranks in aspect_list:
                for i, idx in enumerate(ranks):
                    if idx not in count_dict:
                        count_dict[idx] = 0
                    if idx not in reciprocal_dict:
                        reciprocal_dict[idx] = 0
                    count_dict[idx] += 1
                    reciprocal_dict[idx] += 1 / (i + 1)

            reversed_dict = {}
            for key, value in count_dict.items():
                if value not in reversed_dict:
                    reversed_dict[value] = [key]
                else:
                    reversed_dict[value].append(key)

            sorted_count_dict = {}
            for key in reversed_dict:
                sorted_value = sorted(reversed_dict[key], key=lambda x: (reciprocal_dict[x], -x), reverse=True)
                sorted_count_dict[key] = sorted_value

            reciprocal_list.append(reciprocal_dict)
            bm25_sorted_aspects.append(sorted_count_dict)

        # TODO: change the path
        with open(f"./{output_path}/aspect_sorted_candidate_bm25.pickle", 'wb') as f:
            pickle.dump(bm25_sorted_aspects, f)



    # ======================Filtering======================
    query_rank_tfidf = query_ranking_tfidf
    query_rank_bm25 = query_ranking_bm25
    aspect_rank_tfidf = tfidf_aspect_separate
    aspect_rank_bm25 = bm25_aspect_separate
    aspect_rank_tfidf_sorted = tfidf_sorted_aspects
    aspect_rank_bm25_sorted = bm25_sorted_aspects

    filtered_query_tfidf = get_filtered("query", query_rank_tfidf, temp_dataset, corpus)
    filtered_query_bm25 = get_filtered("query", query_rank_bm25, temp_dataset, corpus)

    filtered_req_tfidf_separated = get_filtered("aspect", aspect_rank_tfidf, temp_dataset, corpus)
    filtered_req_bm25_separated = get_filtered("aspect", aspect_rank_bm25, temp_dataset, corpus)

    filtered_req_tfidf_sorted = get_filtered("sorted", aspect_rank_tfidf_sorted, temp_dataset, corpus)
    filtered_req_bm25_sorted = get_filtered("sorted", aspect_rank_bm25_sorted, temp_dataset, corpus)

    candidate_pool = []
    ground_A = 20
    ground_B = 25
    ground_C = 2
    ground_D = 10
    ground_E = 10

    A = ground_A
    B = ground_B
    C = ground_C
    D = ground_D
    E = ground_E
    

    for i in tqdm(range(len(queries))):
        temp_dict, num_req = create_candidate_per_query(i, A, B, C, D, E, filtered_query_tfidf,
                                                        filtered_req_tfidf_separated, filtered_req_tfidf_sorted,
                                                        filtered_query_bm25, filtered_req_bm25_separated,
                                                        filtered_req_bm25_sorted)
        flag = 0
        while len(temp_dict['TFIDF_and_BM25']) < 80:
            flag += 1
            if flag % 2 == 0:
                A += 1
            else:
                B += 1
            if flag % num_req == 0:
                C += 1
            temp_dict, num_req = create_candidate_per_query(i, A, B, C, D, E, filtered_query_tfidf,
                                                            filtered_req_tfidf_separated, filtered_req_tfidf_sorted,
                                                            filtered_query_bm25, filtered_req_bm25_separated,
                                                            filtered_req_bm25_sorted)
        temp_dict['parameters'] = [A, B, C, D, E]
        candidate_pool.append(temp_dict)

        A = ground_A
        B = ground_B
        C = ground_C
        D = ground_D
        E = ground_E


    # ======================Citation Signals======================
    real_ids = [r['ID'] for r in temp_dataset]
    citation_abs_ids = []

    for r in real_ids:
        temp_list = []
        for abs_id in r:
            temp_list += corpus[abs_id]['incoming_citations']
            temp_list += corpus[abs_id]['outgoing_citations']
        citation_abs_ids.append(temp_list)

    cit_add_to_cand_pool = []
    for i, cand in enumerate(candidate_pool):
        curr_unused = set(cand['unused'])
        curr_citation = set(citation_abs_ids[i])
        cand['citation_related'] = list(curr_unused.intersection(curr_citation))


    # ======================Ada======================
    print("Loading data for ada ...")

    semantic_data = dataset['Corpus']

    print("Complete loading")

    ada_sorted_cand = []
    for i in tqdm(range(dataset_length), desc="Calculating candidates with Ada embedding"):
        try:
            curr_ada_embedding = temp_dataset[i]['ada_embedding']
        except:
            pdb.set_trace()
        cos_sim_dict = {}
        for i in range(len(semantic_data)):
            if 'ada_embedding' in semantic_data[i]:
                cos_sim_dict[i] = cosine_similarity_ada(curr_ada_embedding, semantic_data[i]['ada_embedding'])
        sorted_keys = sorted(cos_sim_dict, key=lambda x: cos_sim_dict[x], reverse=True)
        ada_sorted_cand.append(sorted_keys)

    for i in tqdm(range(dataset_length)):

        ada_list = list(ada_sorted_cand[i])

        curr_cand_pool = candidate_pool[i]
        temp_list = []
        for j in ada_list:

            if (j not in curr_cand_pool['TFIDF_and_BM25']) \
                    and (j not in curr_cand_pool['citation_related']):
                temp_list.append(j)

            if len(temp_list) == 20:
                break
        curr_cand_pool['Ada'] = temp_list

    for i in tqdm(range(dataset_length)):
        candidate_pool[i]['whole_list'] = candidate_pool[i]['TFIDF_and_BM25'] + candidate_pool[i]['citation_related'] + \
                                          candidate_pool[i]['Ada']

    with open(f"./{output_path}/candidate_pool.pickle", 'wb') as f:
        pickle.dump(candidate_pool, f)


# Example python3 create_candidate_pool.py -o candidate_pool_example -q ../benchmarking/dataset/DORIS-MAE_dataset_v1.json -a ../benchmarking/dataset/ada_embedding_for_DORIS-MAE_v1.pickle
