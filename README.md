
# DORIS-MAE: Scientific Document Retrieval using Multi-level Aspect-based Queries



## Table of Contents
* [Dataset](#dataset)
* [Setup](#setup)
* [Experimentation](#experimentation)
  * [Replicating Experiments](#replicating-experiments)
  * [Conducting New Experiments](#conducting-new-experiments)
* [AnnoGPT](#annogpt)
* [Dataset Augmentation](#dataset-augmentation)
* [License](#license)


## Dataset

Welcome to the official repository for the DORIS-MAE project. The dataset associated with this project can be found [here](https://doi.org/10.5281/zenodo.8035110), released under a CC-BY-NC license.

In scientific research, the ability to effectively retrieve relevant documents based on complex, multifaceted queries is critical. Existing evaluation datasets for this task are limited, primarily due to the high costs of annotating resources that capture complex queries.

To address this, we propose a novel task, Scientific Document Retrieval using Multi-level Aspect-based queries (DORIS-MAE), which is designed to handle the complex nature of user queries in scientific research.

We developed a benchmark dataset within the field of computer science, consisting of 50 complex, human-authored primary query cases. For each primary query, we assembled a collection of 100 relevant documents and produced annotated relevance scores for ranking them. Recognizing the significant labor of expert annotation, we also introduce a scalable framework for evaluating the viability of Large Language Models (LLMs) such as ChatGPT-3.5 for expert-level dataset annotation tasks.

The application of this framework to annotate the DORIS-MAE dataset resulted in a 500x reduction in cost, without compromising quality. Furthermore, due to the multi-tiered structure of these primary queries, our DORIS-MAE dataset can be extended to over 4000 sub-query test cases without requiring additional annotation.

We evaluated 10 recent retrieval methods on DORIS-MAE, observing notable performance drops compared to traditional datasets. This highlights DORIS-MAE's challenges and the need for better approaches to handle complex, multifaceted queries in scientific research.

## Setup

To initialize your environment and install necessary packages, use the following commands:

```bash
conda create -n <your_env_name> python=3.8
conda activate <your_env_name>
git clone https://github.com/Real-Doris-Mae/Doris-Mae-Dataset.git
cd Doris-Mae-Dataset/benchmarking
bash run.sh
cd ..
```

## Experimentation

### Replicating Experiments

To reproduce the experiments detailed in the DORIS-MAE paper, please utilize `evluation.py`.
```bash
cd benchmarking
python3 evaluation.py -o <query_option>[query/subquery/aspect] -c <cuda_option> -b <batch_size> -bt <bootstrap_option>
```
For instance, to reproduce the experiments detailed for whole uery level retreival, you can use the line provided below:
```bash
python3 evaluation.py -o "query" -c "0,1,2" -b 200 -bt True
```
To reproduce the experiments with different abstract and query embedding levels, you can edit abstract mode by modifying the 33th and 35th lines, and edit query mode by modifying the 38th and 40th lines in `config.py`.

### Conducting New Experiments

If you're interested in testing the performance of new ranking results based on the metrics used in this repository, please add the new ranking result to the ranking directory and add the appropriate names to `config.py`.

## AnnoGPT

This repository includes code utilizing GPT-3.5 for annotating aspect-abstract pairs. Please note that executing this program requires an OpenAI API key. You can add your OpenAI API key by modifying the 12th line in `GPT_annotation.py`.

To annotate the question pairs, execute the following lines:

```bash
python3 run_annotation.py -m create -d <path_to_the_dataset> -s <annotation_dir> -t 100
cd <annotation_dir>
bash annotation.sh
python3 run_annotation.py -m collect -c <annotation_dir> -o <annotation_result_file>
```

For instance, to annotate the DORIS-MAE dataset, you can use the lines provided below:

```bash
pip install openai
python3 run_annotation.py -m create -d ../benchmarking/dataset/DORIS_MAE_dataset_v0.json -s example_anno_dir -t 100
cd example_anno_dir/
bash annotation.sh
cd ..
python3 run_annotation.py -m collect -c example_anno_dir -o example_output.pickle
```

Please note that `bash annotation.sh` may require several hours to execute. If you wish to run these lines as an example, please change 82754 in line 7 of `annotation.sh` to a smaller number (e.g., 200 or 300).

The annotation results will be stored in `<annotation_result_file>` (e.g., example_output.pickle). In rare cases, you may encounter a result score of -1. For these instances, we recommend manual checking of the result for these pairs.

## Dataset Augmentation

To augment the dataset, please follow the steps below:

1. Construct the queries and aspects.
2. Identify the corpus and corresponding candidate pool for every query.
3. Execute the GPT annotation program.
4. Run the evaluation program.


### Creating New Candidate Pools

To generate new candidate pools for the augmented dataset, utilize the `create_candidate_pool.py` script. It's important to ensure that the `ada-002` embedding for the queries is present in the candidate pool for this script to function correctly. However, even if you decide to construct the candidate pool differently, the other functionalities of this repository will remain operational.

**Steps to Create the Candidate Pool:**

1. Navigate to the directory:
```
   cd create_candidate_pool
```

2. Run the script:
```
python3 create_candidate_pool.py -o <output_dir> -q <path_to_the_dataset> -a <path_to_the_ada_embedding>
```

**Example:**
```
python3 create_candidate_pool.py -o candidate_pool_example -q ../benchmarking/dataset/DORIS-MAE_dataset_v1.json -a ../../benchmarking/dataset/ada_embedding_for_DORIS-MAE_v1.pickle
```

The program is expected to take approximately 90 minutes for a dataset consisting of 50 queries and a 360k corpus. The resulting candidate pool will be saved in the specified `<output_dir>` with the filename `candidate_pool.pickle`.



## License

The code in this project is licensed under the MIT license.

The dataset is under a CC-BY-NC license.

The licenses can be found [here](https://github.com/Real-Doris-Mae/Doris-Mae-Dataset/blob/main/LICENSE.md).
