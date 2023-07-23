# Multimodal Neural Databases

This repository contains the code of the paper "Multimodal Neural Databases", [paper](https://dl.acm.org/doi/10.1145/3539618.3591930) accepted at SIGIR2023.


## Download the dataset and checkpoints

You can find the preprocessed dataset and model checkpoints at [this link](http://www.diag.uniroma1.it/trappolini/support_materials.zip).

Remember to change the corresponding paths in the  `conf/config.yaml` file.

## Install the required libraries

We suggest to use a novel python enviroment before proceding.

Run the command `pip install -r requirements.txt`


## Reproduce the results

To reproduce the results we provide several bash scripts that you can find under the folder `exp`.
Each script is associated to a specific table in the paper (e.g., `tab1.sh`)

### Finetune the clip retriever

To finetune the clip retriever you can use the script `scripts/retrieve_ft.py`

### Finetune the processor

To finetune the processor you can use the script `scripts/processor_ft.py`

### Finetune the stopping algorithm

To finetune the stopping algorithm you can use the script `scripts/stopping_algo.py`



## Citation

If you use this code please cite:

```bibtex
@inproceedings{10.1145/3539618.3591930,
author = {Trappolini, Giovanni and Santilli, Andrea and Rodol\`{a}, Emanuele and Halevy, Alon and Silvestri, Fabrizio},
title = {Multimodal Neural Databases},
year = {2023},
isbn = {9781450394086},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539618.3591930},
doi = {10.1145/3539618.3591930},
abstract = {The rise in loosely-structured data available through text, images, and other modalities has called for new ways of querying them. Multimedia Information Retrieval has filled this gap and has witnessed exciting progress in recent years. Tasks such as search and retrieval of extensive multimedia archives have undergone massive performance improvements, driven to a large extent by recent developments in multimodal deep learning. However, methods in this field remain limited in the kinds of queries they support and, in particular, their inability to answer database-like queries. For this reason, inspired by recent work on neural databases, we propose a new framework, which we name Multimodal Neural Databases (MMNDBs). MMNDBs can answer complex database-like queries that involve reasoning over different input modalities, such as text and images, at scale. In this paper, we present the first architecture able to fulfill this set of requirements and test it with several baselines, showing the limitations of currently available models. The results show the potential of these new techniques to process unstructured data coming from different modalities, paving the way for future research in the area.},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2619–2628},
numpages = {10},
keywords = {databases, neural networks, multimedia information retrieval},
location = {Taipei, Taiwan},
series = {SIGIR '23}
}
```
