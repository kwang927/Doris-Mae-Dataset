import json
import pickle
import argparse
from annotation_util import *

# This script is used to annotate question pairs from an input dataset.
# It operates in two modes: 'create' and 'collect'.
# In 'create' mode, it reads a dataset and prepares a structure (shell) for annotation.
# In 'collect' mode, it collects GPT annotation results from the directory specified.
# The program requires multiple command-line arguments to function correctly, as described below.

# The arguments are:
    # '--mode': Specifies the operation mode. It must be 'create' or 'collect'.
    # '--dataset': (For 'create' mode) Path to the dataset to be annotated.
    # '--store': (For 'create' mode) Path where the annotation structure should be stored.
    # '--thread': Number of threads used for annotation. The suggested values are 5, 10, 20, 25, 50, or 100.
    # '--collect': (For 'collect' mode) Path where the GPT annotation results are located.
    # '--output': (For 'collect' mode) Path where the parsed GPT annotations should be stored.
    # The parser will automatically generate help and usage messages and issue errors when users give the program invalid arguments.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='annotate question pairs from the input dataset',
        epilog='Created by Doris Mae'
    )

    parser.add_argument('-m', '--mode', required=True, help="Mode of the program, must be one of ['create', 'collect']")
    parser.add_argument('-d', '--dataset', default=None, help='path to the dataset to annotate for create mode')
    parser.add_argument('-s', '--store', default=None, help='path to store the current annotation for create mode')
    parser.add_argument('-t', '--thread', default=100,
                        help='Number of threads used during the annotation process (suggested to be one of [5, 10 ,20, 25, 50, 100])')
    parser.add_argument('-c', '--collect', default=None, help='path to the gpt annotation results for collect mode')
    parser.add_argument('-o', '--output', default='parsed_annotation.pickle', help = 'output path for parsed gpt annotations for collect mode')

    args = parser.parse_args()
    if args.mode == 'create':
        path_for_dataset = args.dataset
        path_for_annotation = args.store
        num_of_threads = args.thread
        if path_for_dataset is None or path_for_annotation is None:
            raise RuntimeError("Missing argument for create mode")

        print(f"Reading dataset from {path_for_dataset} ...")
        with open(path_for_dataset, "r") as f:
            data = json.load(f)
        create_question_pair_set_and_shell_for_annotation(data, num_of_threads, path_for_annotation)
        print(f"The question pairs and shell for annotation are stored at {path_for_annotation}")

    elif args.mode == 'collect':
        path_for_annotation = args.collect
        path_to_output = args.output
        if path_for_annotation is None:
            raise RuntimeError("Missing argument for collect mode")

        dir_lists = get_subfolders(path_for_annotation)
        gpt_annotation_results = collect_gpt_results(dir_lists)
        with open(path_to_output, 'wb') as f:
            pickle.dump(gpt_annotation_results, f)

        print(f"Parsed GPT annotation has been stored to {path_to_output}.")

    else:
        raise RuntimeError("Mode is not 'create' or 'collect'.")
