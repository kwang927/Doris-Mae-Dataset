import numpy as np
from scipy.stats import spearmanr, pearsonr
import pdb
from sklearn.metrics import precision_score, recall_score, f1_score


def f1_stats(x, y):
    """
    Input: x, y are two list of scores with same size

    Output: The macro (taking all classes as equally important) precision, recall and f1 score between the two lists
    """

    if len(x) != len(y):
        return

    true_labels = y
    predicted_labels = x
    precision = precision_score(true_labels, predicted_labels, average="macro")
    recall = recall_score(true_labels, predicted_labels, average="macro")
    f1 = f1_score(true_labels, predicted_labels, average="macro")

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return precision, recall, f1


def calculate_spearman_correlation(x, y):
    """
    Calculates the Spearman's correlation coefficient between two arrays.

    Args:
        x (array-like): First array of values.
        y (array-like): Second array of values.

    Returns:
        float: The Spearman's correlation coefficient.
    """
    # # Convert the arrays to numpy arrays if they are not already

    x = np.array(x)
    y = np.array(y)

    # Calculate the Spearman's correlation coefficient
    correlation, _ = spearmanr(x, y)

    print(f"Spearman's Rho is {correlation}")

    return correlation


def exact_agreement(x, y, Print=False):
    """
    Input: x, y are two list of scores with same size
           print is a boolean that controls whether exact agreement rate is printed

    Output: The list of index that are not exactly matching, and the exact agreement rate
    """

    count = 0
    list_of_error = []
    for i in range(len(x)):
        if x[i] == y[i]:
            count += 1
        else:
            list_of_error.append(i)
    if Print:
        print(f"The percentage of exact agreement is {count / len(x)}")
    return list_of_error, count / len(x)


# def inexact_agreement(x, y, Print=False):
#     count = 0
#     list_of_error = []
#     for i in range(len(x)):
#         if abs(x[i] - y[i]) < 2:
#             count += 1
#         else:
#             list_of_error.append(i)
#     if Print:
#         print(f"The percentage of inexact agreement is {count / len(x)}")
#     return list_of_error


def find_distribution(x, name="model"):
    """
    Input: a list of scores

    Output: A dictionary that records the distribution of the scores
            The keys will be 0, 1, and 2 and the value would be the ratio of the key in the input list
    """
    result_x = {0: 0, 1: 0, 2: 0}

    for i in x:
        result_x[i] += 1 / len(x)
    for i in sorted(list(result_x.keys())):
        result_x[i] = round(result_x[i], 2)

    print(f"The distribution of {name} is {result_x}")

    return result_x
