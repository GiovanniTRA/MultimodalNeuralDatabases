# Multimodal Neural Databases

This repository contains the code of the paper "Multimodal Neural Databases".


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
@article{trappolini2023multimodal,
  title={Multimodal Neural Databases},
  author={Trappolini, Giovanni and Santilli, Andrea and Rodol{\`a}, Emanuele and Halevy, Alon and Silvestri, Fabrizio},
  journal={arXiv preprint arXiv:2305.01447},
  year={2023}
}
```