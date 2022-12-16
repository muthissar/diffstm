# Differentiable Short-Term Model (DSTM)
This repository holds code for [Discrete Short-Term Models](https://doi.org/10.5334/tismir.123):
> ## Abstract 
> As pieces of music are usually highly self-similar, online-learning short-term models
are well-suited for musical sequence prediction tasks. Due to their simplicity and
interpretability, Markov chains (MCs) are often used for such online learning, with
Prediction by Partial Matching (PPM) being a more sophisticated variant of simple
MCs. PPM, also used in the well-known IDyOM model, constitutes a variable-order MC
that relies on exact matches between observed *n*-grams and weights more recent
events higher than those further in the past. We argue that these assumptions are
limiting and propose the Differentiable Short-Term Model (DSTM) that is not limited
to exact matches of *n*-grams and can also learn the relative importance of events.
During (offline-)training, the DSTM learns representations of *n*-grams that are useful
for constructing fast weights (that resemble an MC transition matrix) in online learning
of *intra-opus* pitch prediction. We propose two variants: the Discrete Code Short-
Term Model and the Continuous Code Short-Term Model. We compare the models to
different baselines on the [*“TheSession“*](https://github.com/IraKorshunova/folk-rnn/) dataset and find, among other things, that
the Continuous Code Short-Term Model has a better performance than Prediction by
Partial Matching, as it adapts faster to changes in the data distribution. We perform
an extensive evaluation of the models, and we discuss some analogies of DSTMs
with linear transformers.
## Install
### pytorch
It is recommended to obtain pytorch using the official [install instructions](https://pytorch.org/get-started).

### dstm
Install the dstm package by running:

``
pip install -e .
``
## `dstm.py`
`dstm.py` is used for training and evaluating models. The program is build with [pytorch-ligthing](https://github.com/Lightning-AI/lightning) and supports the arguments of [pytorch_ligthing.trainer.trainer.Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags) (e.g, (multiple) GPUs)).


For information on program arguments run:

``
python dstm.py --help
``


### Training
Training is on by default but can be switched off by using `--skip_train`. For instance, to reproduce the best performing Continous Code Short-Term Model from the paper run: 

``
python dstm.py --batch_size=16 --dropout=0.1 --encoder_output_dim=512 --early_stopping --filters 512 512 512 512 512 512 512 512 --lr=0.0001 --activation=selu dstm --short_term_method=elu
``  

To speed up training using GPU acceleration consider adding `` --accelerator=gpu --devices=0,1,2,3 --strategy=ddp`` and changing ``--batch_size=4``.  

When experiencing memory issues, try lowering the batch size (e.g., `--batch_size=4`).

### Evaluation
Evaluation is on by default but can be switched off by using `--skip_test`. We provide two pretrained models from the paper: `out/session/model/ccstm.ckpt` and `out/session/model/dcstm.ckpt`. These are downloaded from the CDN on first run. The performance of the checkpoints can be evaluated by:  
``
python dstm.py --batch_size=16 --skip_train --checkpoint="out/session/model/ccstm.ckpt" --encoder_output_dim=512 --no_log dstm 
``

## Issues
Feel free to open an issue in case something does not work.
