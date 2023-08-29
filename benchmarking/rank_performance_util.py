import math
import numpy as np
import pdb
import random
from tqdm import tqdm

# Precision at k: It measures the proportion of recommended items in the top-k set that are relevant
def precision_at_k(relevance_list, ranking_result, k=20):
    """
    Parameters:
        relevance_list (list): List of relevant document ids.
        ranking_result (list): List that shows the ranking of the documents.
        k (int): The 'k' in 'P@k'. Default is 20.

    Returns:
        float: The precision at k of the ranking result.
    """
    ranking_result = ranking_result[:k]
    relevance_set = set(relevance_list)
    ranking_set = set(ranking_result)
    inter_size = len(relevance_set.intersection(ranking_set))
    return inter_size / k


# Recall at k: It measures the proportion of relevant items found in the top-k recommendations
def recall_at_k(relevance_list, ranking_result, k=20):
    """
    Parameters:
        relevance_list (list): List of relevant document ids.
        ranking_result (list): List that shows the ranking of the documents.
        k (int): The 'k' in 'R@k'. Default is 20.

    Returns:
        float: The recall at k of the ranking result.
    """
    ranking_result = ranking_result[:k]
    relevance_set = set(relevance_list)
    ranking_set = set(ranking_result)

    inter_size = len(relevance_set.intersection(ranking_set))
    return inter_size / len(relevance_list)


def r_precision(relevance_list, ranking_result):
    """
    Parameters:
        relevance_list (list): List of relevant document ids.
        ranking_result (list): List that shows the ranking of the documents.

    Returns:
        float: R-precision of the ranking result.
    """
    relevance_len = len(relevance_list)
    ranking_result = ranking_result[:relevance_len]
    relevance_set = set(relevance_list)
    ranking_set = set(ranking_result)

    inter_size = len(relevance_set.intersection(ranking_set))
    return inter_size / relevance_len


# Normalized Discounted Cumulative Gain (NDCG): A measure of ranking quality. It uses the graded relevance of a query result set and discounts the relevance of documents lower down in the result list.
def ndcg_normal(ground_truth_ranking, ground_truth_score, ranking_result, p):
    """
    Parameters:
        ground_truth_ranking (list): List that shows the correct ranking of the documents.
        ground_truth_score (list): List that shows the correct score of the documents.
        ranking_result (list): List that shows the ranking of the documents.
        p (float): A float between 0 and 1

    Returns:
        float: NDCG of the ranking result given the ground truth ranking and score.
    """

    consider_length = int(p * len(ranking_result))
    dcg = 0
    for ind in range(consider_length):
        curr_id = ranking_result[ind]
        pos_in_ground_truth = ground_truth_ranking.index(curr_id)
        rel_i = ground_truth_score[pos_in_ground_truth]
        dcg += rel_i / math.log2(ind + 2)
    idcg = 0
    for ind in range(consider_length):
        rel_i = ground_truth_score[ind]
        idcg += rel_i / math.log2(ind + 2)
    return dcg / idcg


# Alternate version of NDCG: Similar to NDCG, but this version uses a exponential gain to emphasise the importance of relevance.
def ndcg_exp(ground_truth_ranking, ground_truth_score, ranking_result, p):
    """
    Parameters:
        ground_truth_ranking (list): List that shows the correct ranking of the documents.
        ground_truth_score (list): List that shows the correct score of the documents.
        ranking_result (list): List that shows the ranking of the documents.
        p (float): A float between 0 and 1

    Returns:
        float: NDCG (exponential) of the ranking result given the ground truth ranking and score.
    """
    consider_length = int(p * len(ranking_result))
    dcg = 0
    for ind in range(consider_length):
        curr_id = ranking_result[ind]
        pos_in_ground_truth = ground_truth_ranking.index(curr_id)
        rel_i = ground_truth_score[pos_in_ground_truth]
        dcg += (2 ** rel_i) / math.log2(ind + 2)
    idcg = 0
    for ind in range(consider_length):
        rel_i = ground_truth_score[ind]
        idcg += (2 ** rel_i) / math.log2(ind + 2)
    return dcg / idcg


# Reciprocal Rank: The reciprocal of the rank of the first relevant document
def reciprocal_rank(ground_truth_rank, ground_truth_score, ranking_result, k):
    """
    Parameters:
        ground_truth_rank (list): List that shows the correct ranking of the documents.
        ground_truth_score (list): List that shows the correct score of the documents.
        ranking_result (list): List that shows the ranking of the documents.
        k (int): An integer between 1 and the length of the ranking result

    Returns:
        float: Reciprocal rank of the ranking result.
    """

    visible_ranking_result = ranking_result[:k]
    max_score = ground_truth_score[0]
    for ind in range(len(ground_truth_rank)):
        if ground_truth_score[ind] < max_score:
            break
    relevance_docs = ground_truth_rank[:ind]
    for i in range(len(visible_ranking_result)):
        if visible_ranking_result[i] in relevance_docs:
            return 1 / (i + 1)
    return 0


def average_precision(relevance_list, ranking_result):
    """
    Parameters:
        relevance_list (list): List of relevant document ids.
        ranking_result (list): List that shows the ranking of the documents.

    Returns:
        float: R-precision of the ranking result.
    """
    relevant_docs = set(relevance_list)
    num_relevant_docs = len(relevant_docs)
    if num_relevant_docs == 0:
        return 0.0
    cum_sum_precisions = 0.0
    num_hits = 0
    for i, doc_id in enumerate(ranking_result):
        if doc_id in relevant_docs:
            num_hits += 1
            precision_at_i = num_hits / (i + 1.0)
            cum_sum_precisions += precision_at_i
    ret = cum_sum_precisions / num_relevant_docs
    return ret


def sort_dict_by_values(dictionary):
    sorted_dict = dict(sorted(dictionary.items(), key=lambda x: x[1], reverse=True))
    return sorted_dict


def generate_random_rank(rank_template, seed):
    random.seed(seed)
    randomized_rank = []
    for d in rank_template:
        temp_rank = d['index_rank'].copy()
        random.shuffle(temp_rank)
        randomized_rank.append({'index_rank': temp_rank})

    return randomized_rank


def get_random_baseline(gpt_result, rank, test_suites, trials=1000):
    '''
    Parameters: 
           gpt_result: List of results from GPT model.
           test_suites: test metrics
           rank: A result rank dictionary for performing random baseline
           trials: number of trials for performing random baseline
    Return:
          returning rank, rank is a list of dictionary, the position index corresponds to the number of query, and in each dicitonary, the only key is index_rank, the value is a list of paper abstarct ids sorted in descending order w.r.t. their relevance of the given query, based on the current random basline. 
    '''
    metric_result_dict = {}
    for i in tqdm(range(trials), desc="Generating random trails", leave=False):
        curr_seed = i
        curr_random_rank = generate_random_rank(rank, curr_seed)
        for test_name in test_suites.keys():
            for k in test_suites[test_name]:
                value = get_metric(gpt_result, curr_random_rank, test_name, k)
                if (test_name, k) not in metric_result_dict.keys():
                    metric_result_dict[(test_name, k)] = [value]
                else:
                    metric_result_dict[(test_name, k)].append(value)

    return metric_result_dict


def print_random_baseline(test_suites, metric_result_dict):
    '''
    Parameters: 
           test_suites: test metrics
           metric_result_dict: A result rank dictionary for random baseline
    '''
    model_name = "random"
    print()
    print("Random metrics:")
    for test_name in test_suites.keys():
        for k in test_suites[test_name]:
            value = np.mean(metric_result_dict[(test_name, k)])
            if k != None:
                print(f"{model_name} : {test_name}@{k}  : {value}%")
            else:
                print(f"{model_name} : {test_name}  : {value}%")

    print('=' * 100)


def get_metric(gpt_result, method_rank, metric_name, k=None):
    """
    This function calculates various IR metrics including Recall, R Precision, NDCG, NDCG_exp, MRR, and MAP.

    Parameters:
        gpt_result (list): List of results from GPT model.
        method_rank (list): List of ranked document ids.
        metric_name (str): The name of the metric to be calculated.
        k (int, optional): An optional parameter for some metrics (default is None).

    Returns:
        float: Mean of the computed metric across all queries.

    Raises:
        RuntimeError: If 'k' is not provided where required, or metric_name is not recognized.
    """
    assert len(gpt_result) == len(method_rank), f"{len(gpt_result)}; {len(method_rank)}"
    removed_list = [205580, 346695, 346826, 346836, 28674]
    
    # threshold for determining relevant list
    threshold = 1.0
    
    relevance_lists = []
    original_gpt_result = []

    for i in gpt_result:
        curr_original_gpt_result = {}
        for abs_id in i.keys():
            curr_original_gpt_result[abs_id] = i[abs_id][0]

        curr_original_gpt_result = sort_dict_by_values(curr_original_gpt_result)
        original_gpt_result.append(curr_original_gpt_result)

    for i in gpt_result:
        curr_relevance_list = []
        for abs_id in i.keys():
            curr_normalized_score = i[abs_id][1]
            if curr_normalized_score >= threshold:
                curr_relevance_list.append(abs_id)

            curr_relevance_list = [c for c in curr_relevance_list if c not in removed_list]
        relevance_lists.append(curr_relevance_list)

    if metric_name == "recall":
        if k is None or k <= 0:
            raise RuntimeError("k should be positive for recall")

        ret_list = []

        for ind in range(len(method_rank)):
            curr_relevance_list = relevance_lists[ind]
            curr_ranking_result = method_rank[ind]['index_rank']
            if len(curr_relevance_list) == 0:
                continue
            ret_list.append(recall_at_k(curr_relevance_list, curr_ranking_result, k))
        return np.mean(ret_list) * 100

    elif metric_name == 'r_precision':
        ret_list = []

        for ind in range(len(method_rank)):
            curr_relevance_list = relevance_lists[ind]
            curr_ranking_result = method_rank[ind]['index_rank']
            if len(curr_relevance_list) == 0:
                continue
            ret_list.append(r_precision(curr_relevance_list, curr_ranking_result))
        return np.mean(ret_list) * 100

    elif metric_name == 'ndcg':
        ret_list = []
        if k is None or k <= 0 or k > 1:
            raise RuntimeError("k should be between 0 and 1 for NDCG")

        for ind in range(len(method_rank)):
            curr_gpt_rank = list(original_gpt_result[ind].keys())
            curr_gpt_score = list(original_gpt_result[ind].values())
            curr_ranking_result = method_rank[ind]['index_rank']
            p = k

            ret_list.append(ndcg_normal(curr_gpt_rank, curr_gpt_score, curr_ranking_result, p))
        return np.mean(ret_list) * 100

    elif metric_name == 'ndcg_exp':
        ret_list = []
        if k is None or k <= 0 or k > 1:
            raise RuntimeError("k should be between 0 and 1 for NDCG")

        for ind in range(len(method_rank)):
            curr_gpt_rank = list(original_gpt_result[ind].keys())
            curr_gpt_score = list(original_gpt_result[ind].values())
            curr_ranking_result = method_rank[ind]['index_rank']
            p = k

            ret_list.append(ndcg_exp(curr_gpt_rank, curr_gpt_score, curr_ranking_result, p))
        return np.mean(ret_list) * 100

    elif metric_name == 'mrr':
        ret_list = []
        if k is None or k <= 0:
            raise RuntimeError("k should be positive for MRR")

        for ind in range(len(method_rank)):
            curr_gpt_rank = list(original_gpt_result[ind].keys())
            curr_gpt_score = list(original_gpt_result[ind].values())
            curr_ranking_result = method_rank[ind]['index_rank']
            ret_list.append(reciprocal_rank(curr_gpt_rank, curr_gpt_score, curr_ranking_result, k))
        return np.mean(ret_list) * 100

    elif metric_name == 'map':
        ret_list = []

        for ind in range(len(method_rank)):
            curr_relevance_list = relevance_lists[ind]
            curr_ranking_result = method_rank[ind]['index_rank']
            if len(curr_relevance_list) == 0:
                continue
            ret_list.append(average_precision(curr_relevance_list, curr_ranking_result))
        return np.mean(ret_list) * 100

    elif metric_name == 'binary_acc':

        ret_list = []

        for ind in range(len(method_rank)):

            curr_relevance_list = relevance_lists[ind]
            curr_ranking_result = method_rank[ind]['index_rank']

            if len(curr_relevance_list) <= 1:
                continue
            curr_gpt_rank = list(original_gpt_result[ind].keys())
            ret_list.append(binary_classification(curr_relevance_list, curr_ranking_result, curr_gpt_rank))
        return np.mean(ret_list) * 100

    else:
        raise RuntimeError("metric_name should be one of ['recall', 'r_precision', 'ndcg', 'ndcg_exp', 'mrr', 'map']")
        
def get_full_metric_results(gpt_result, method_rank, metric_name, k=None):
    """
    This function calculates various IR metrics including Recall, R Precision, NDCG, NDCG_exp, MRR, and MAP.

    Parameters:
        gpt_result (list): List of results from GPT model.
        method_rank (list): List of ranked document ids.
        metric_name (str): The name of the metric to be calculated.
        k (int, optional): An optional parameter for some metrics (default is None).

    Returns:
        float: Mean of the computed metric across all queries.

    Raises:
        RuntimeError: If 'k' is not provided where required, or metric_name is not recognized.
    """
    assert len(gpt_result) == len(method_rank)
    removed_list = [205580, 346695, 346826, 346836, 28674]
    
    # threshold for determining relevant list
    threshold = 1.0
    
    relevance_lists = []
    original_gpt_result = []

    for i in gpt_result:
        curr_original_gpt_result = {}
        for abs_id in i.keys():
            curr_original_gpt_result[abs_id] = i[abs_id][0]

        curr_original_gpt_result = sort_dict_by_values(curr_original_gpt_result)
        original_gpt_result.append(curr_original_gpt_result)

    for i in gpt_result:
        curr_relevance_list = []
        for abs_id in i.keys():
            curr_normalized_score = i[abs_id][1]
            if curr_normalized_score >= threshold:
                curr_relevance_list.append(abs_id)

            curr_relevance_list = [c for c in curr_relevance_list if c not in removed_list]
        relevance_lists.append(curr_relevance_list)

    if metric_name == "recall":
        if k is None or k <= 0:
            raise RuntimeError("k should be positive for recall")

        ret_list = []

        for ind in range(len(method_rank)):
            curr_relevance_list = relevance_lists[ind]
            curr_ranking_result = method_rank[ind]['index_rank']
            if len(curr_relevance_list) == 0:
                continue
            ret_list.append(recall_at_k(curr_relevance_list, curr_ranking_result, k))
        return ret_list

    elif metric_name == 'r_precision':
        ret_list = []

        for ind in range(len(method_rank)):
            curr_relevance_list = relevance_lists[ind]
            curr_ranking_result = method_rank[ind]['index_rank']
            if len(curr_relevance_list) == 0:
                continue
            ret_list.append(r_precision(curr_relevance_list, curr_ranking_result))
        return ret_list

    elif metric_name == 'ndcg':
        ret_list = []
        if k is None or k <= 0 or k > 1:
            raise RuntimeError("k should be between 0 and 1 for NDCG")

        for ind in range(len(method_rank)):
            curr_gpt_rank = list(original_gpt_result[ind].keys())
            curr_gpt_score = list(original_gpt_result[ind].values())
            curr_ranking_result = method_rank[ind]['index_rank']
            p = k

            ret_list.append(ndcg_normal(curr_gpt_rank, curr_gpt_score, curr_ranking_result, p))
        return ret_list

    elif metric_name == 'ndcg_exp':
        ret_list = []
        if k is None or k <= 0 or k > 1:
            raise RuntimeError("k should be between 0 and 1 for NDCG")

        for ind in range(len(method_rank)):
            curr_gpt_rank = list(original_gpt_result[ind].keys())
            curr_gpt_score = list(original_gpt_result[ind].values())
            curr_ranking_result = method_rank[ind]['index_rank']
            p = k

            ret_list.append(ndcg_exp(curr_gpt_rank, curr_gpt_score, curr_ranking_result, p))
        return ret_list

    elif metric_name == 'mrr':
        ret_list = []
        if k is None or k <= 0:
            raise RuntimeError("k should be positive for MRR")

        for ind in range(len(method_rank)):
            curr_gpt_rank = list(original_gpt_result[ind].keys())
            curr_gpt_score = list(original_gpt_result[ind].values())
            curr_ranking_result = method_rank[ind]['index_rank']
            ret_list.append(reciprocal_rank(curr_gpt_rank, curr_gpt_score, curr_ranking_result, k))
        return ret_list

    elif metric_name == 'map':
        ret_list = []

        for ind in range(len(method_rank)):
            curr_relevance_list = relevance_lists[ind]
            curr_ranking_result = method_rank[ind]['index_rank']
            if len(curr_relevance_list) == 0:
                continue
            ret_list.append(average_precision(curr_relevance_list, curr_ranking_result))
        return ret_list

    elif metric_name == 'binary_acc':

        ret_list = []

        for ind in range(len(method_rank)):

            curr_relevance_list = relevance_lists[ind]
            curr_ranking_result = method_rank[ind]['index_rank']

            if len(curr_relevance_list) <= 1:
                continue
            curr_gpt_rank = list(original_gpt_result[ind].keys())
            ret_list.append(binary_classification(curr_relevance_list, curr_ranking_result, curr_gpt_rank))
        return ret_list

    else:
        raise RuntimeError("metric_name should be one of ['recall', 'r_precision', 'ndcg', 'ndcg_exp', 'mrr', 'map']")
