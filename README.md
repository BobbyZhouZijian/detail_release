# Generalizing Influence to Demontration Attribution in In-context Learning

## [Paper link](https://arxiv.org/abs/2405.14899) (Our paper is accepted to NeurIPS 2024!)

**Abstract**:
In-context learning (ICL) allows transformer-based language models that are pre-trained on general text to quickly learn a specific task with a few "task demonstrations" without updating their parameters, significantly boosting their flexibility and generality. ICL possesses many distinct characteristics from conventional machine learning, thereby requiring new approaches to interpret this learning paradigm. Taking the viewpoint of recent works showing that transformers learn in context by formulating an internal optimizer, we propose an influence function-based attribution technique, DETAIL, that addresses the specific characteristics of ICL. We empirically verify the effectiveness of our approach for demonstration attribution while being computationally efficient. Leveraging the results, we then show how DETAIL can help improve model performance in real-world scenarios through demonstration reordering and curation. Finally, we experimentally prove the wide applicability of DETAIL by showing our attribution scores obtained on white-box models are transferable to black-box models in improving model performance.


## Usage
- For running the MNIST experiment, install the conda environment `jax_env.yml` and run `CUDA_VISIBLE_DEVICES=0 python mnist_data_attr.py`
- For running the LLM experiments, install the conda environment `torch_env.yml` and run the programs described in `script.sh`
- To analyze the results, use `analyzer.ipynb`.

## Acknowledgement
Parts of the code are referenced from the following repositories
- [**Transformers Learn In-context by Gradient Descent**](https://github.com/google-research/self-organising-systems/tree/master/transformers_learn_icl_by_gd)
- [**Label Words are Anchors**](https://github.com/lancopku/label-words-are-anchors)
- [**Use Your INSTINCT: Instruction Optimization Using Neural Bandits Coupled with Transformers**](https://github.com/xqlin98/INSTINCT)

## Credit our work
If you find our work interesting, please star our repository. If you wish to cite our paper, you may use the following citation format
```
@inproceedings{zhou2024detail,
      title={DETAIL: Task DEmonsTration Attribution for Interpretable In-context Learning}, 
      author={Zijian Zhou and Xiaoqiang Lin and Xinyi Xu and Alok Prakash and Daniela Rus and Bryan Kian Hsiang Low},
      year={2024},
      booktitle={Advances in Neural Information Processing Systems}
}
```
