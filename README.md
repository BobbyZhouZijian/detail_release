# Gneralizing Influence to In-context Learning

## Usage
- For running the MNIST experiment, install the conda environment `jax_env.yml` and run `CUDA_VISIBLE_DEVICES=0 python mnist_data_attr.py`
- For running the LLM experiments, install the conda environment `torch_env.yml` and run the programs described in `script.sh`
- To analyze the results, use `analyzer.ipynb`.

## Acknowledgement
Parts of the code are referenced from the following repositories
- [**Transformers Learn In-context by Gradient Descent**](https://github.com/google-research/self-organising-systems/tree/master/transformers_learn_icl_by_gd)
- [**Label Words are Anchors**](https://github.com/lancopku/label-words-are-anchors)
- [**Use Your INSTINCT: Instruction Optimization Using Neural Bandits Coupled with Transformers**](https://github.com/xqlin98/INSTINCT)