import pickle
import os
import shutil
import sys
sys.path.append("../benchmarking")
from dataset_util import *


def create_question_pair_set_and_shell_for_annotation(data, num_of_threads, output_path):
    """
    Input: the entire dataset, the dataset needs to have at least
            4 keys: 'aspect2aspect_id', 'aspect_id2aspect', 'Query', 'Corpus'.

    Output: no output, but will create a list of question pairs and a shell can can be used to annotate
            the question pairs in the output_path

    """

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    question_pairs = get_question_pairs(data)
    question_pair_path = f"{output_path}/question_pairs.pickle"
    with open(question_pair_path, 'wb') as f:
        pickle.dump(question_pairs, f)

    full_length = len(question_pairs)
    step_size = 100

    shell_str = f"""
#!/bin/bash

echo "shell starting ..."

# Loop variables start from 0 to {full_length} with step {num_of_threads}
for ((i=0; i<={full_length}; i+={num_of_threads}))
do
    echo "sleeping 30s, the program is running normally"
    sleep 30s
    start=$i
    end=$((i+{num_of_threads}))

    # Run the command with a 2-minute timeout
    if ! timeout 3m python ../GPT_annotation.py -q question_pairs.pickle -o annotations -p ../prompt_config/prompt_config.pickle -c ht -t {num_of_threads} -s $start -e $end
    then
        # If the command times out (exits with a status greater than 128), sleep for 2 minutes
        echo "The program timed out, sleeping for 2 minutes..."
        sleep 2m
    fi
done
"""
    with open(f"{output_path}/annotation.sh", 'w') as f:
        f.write(shell_str)
    # print(shell_str)


def parse_answer(combined_output):
    """
    Input: combined_output: The list of list of dictionary that records the conversation with ChatGPT for annotation
    Output: The parsed result from the conversations. The normal scores are 0-2. If the conversation does not have
    sufficient information, -1 will be given and the user need to check the score.
    """

    score_list = []
    for ind, i in enumerate(combined_output):
        curr_final_answer = i[-1]['content']

        tmp = []

        if "AGREE" in curr_final_answer and "DISAGREE" not in curr_final_answer:
            tmp.append(3)
        if "DISPUTE" in curr_final_answer:
            tmp.append(2)
        if "DISAGREE" in curr_final_answer:
            tmp.append(1)
        if "CONCUR" in curr_final_answer:
            tmp.append(4)
        if len(tmp) != 1:
            score_list.append(-1)
        else:
            score_list.append(tmp[0])

    score_map = {-1: -1, 1: 0, 2: 1, 3: 1, 4: 2}
    ret = [score_map[i] for i in score_list]

    return ret


def collect_gpt_results(dir_list):
    """
    Input: dir_list: The list of directory names that the results of ChatGPT's annotation conversations
    Output: A dictionary that stores the question pair id and the parsed result of ChatGPT's annotation.
            The key will be a tuple, the first position is the id of the aspect, and the second position is the id of
            the abstract. The value will be the parsed ChatGPT annotation.
    """

    ret = []
    for d in dir_list:
        tmp_list = []
        curr_combined_result_path = d + "/combined_result.json"
        curr_index_list_path = d + "/index_of_results.pickle"

        with open(curr_combined_result_path, 'r') as f:
            curr_combined_result = json.load(f)

        with open(curr_index_list_path, 'rb') as f:
            curr_index_list = pickle.load(f)

        parsed_scores = parse_answer(curr_combined_result)
        assert len(parsed_scores) == len(curr_index_list)

        for i in range(len(parsed_scores)):
            index_tuple = curr_index_list[i]
            gpt_response = curr_combined_result[i]
            tmp_list.append(
                {"aspect_id": index_tuple[0], "abstract_id": index_tuple[1], "gpt_response": gpt_response[1]['content'],
                 "score": parsed_scores[i]})

        ret += tmp_list

    return ret

def get_subfolders(folder_path):
    """"
    Input: the name of the directory that contains the results of the gpt annotations
    Output: A list of strings for the path of the folders under the input directory 
    """

    folders = []
    for entry in os.scandir(folder_path+"/annotation_results"):
        if entry.is_dir():
            folders.append(entry.name)

    path_to_folders = [folder_path + "/annotation_results/" + i for i in folders]
    return path_to_folders
