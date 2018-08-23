# Tensorflow Implementation of "Language Modeling with Gated Convolutional Networks"
In this project, I implemented Facebook AI Research Lab's paper: [Language Modeling with Gated Convolutional Networks](https://arxiv.org/pdf/1612.08083.pdf).
## Requirements
- Python 2.7
- Tensorflow >=1.8.0
- numpy
- tqdm
- codecs
## Execution
- STEP 0, Download [Google 1-Billion-Word Dataset](www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz)
You need to store the dataset in './datas'.
- STEP 1, Adjust hyper parameters in `hyperparams.py`.
- STEP 2, Run `python train.py` for training model. models are saved in `./logdir/` directory.

## Notes
- I used NCE loss instead of Adaptive Softmax.
- I used zero-padding for fixing the length of sentences and preventing the kernels from seeing future context.
## Currently
- Training is too slow and unstable. (I tested only with CPU)
- Training loss is not converged.
## References
- https://github.com/anantzoid/Language-Modeling-GatedCNN
