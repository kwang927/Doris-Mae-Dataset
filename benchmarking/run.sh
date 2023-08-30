#!/bin/bash
pip install -r pre_requirements.txt #installing requirements
python3 modifications/combine_lfs.py

wget -P dataset https://zenodo.org/record/8299749/files/DORIS-MAE_dataset_v1.json # downloading doris mae dataset v1.
wget -P dataset https://zenodo.org/record/8299749/files/ada_embedding_for_DORIS-MAE_v1.pickle # downloading ada embedding of the dataset.


git clone https://github.com/stanford-futuredata/ColBERT.git # downloading colbertv2 model
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz
tar -xzf colbertv2.0.tar.gz
mv colbertv2.0 ColBERT/
rm colbertv2.0.tar.gz



git clone https://github.com/allenai/aspire.git
cp -f modifications/ex_aspire_consent.py aspire/examples/ex_aspire_consent.py # enabling cuda option in aspire model
cp modifications/aspire_distance.py aspire/

pip install -r post_requirements.txt #installing requirements for colbert and aspire
