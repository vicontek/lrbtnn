# Bayesian Tensorized Neural Networks with Automatic RankSelection

The repository reproduces ideas from https://arxiv.org/abs/1905.10478 with some additional experiments.

## Requirements

To install all required packages, run the following command:

```bash
pip install -r requirements.txt
```

## How to use

The **Generative Latent Flow** (GLF) is an algorithm for generative modeling of the data distribution. 
One could use it to generate images. 

### Experiments

We perfromed a set of experiments aimed to reproduce automatic low rank selection and MAP-training. 
Also, we tested our own ideas, described in more details in the report.

```TTModel_exps.ipynb``` contains common training procedure for model with two TT-FC layers with default manually set ranks.

```MAP_lambdas_training.ipynb``` contains loss functions and training procedure for general MAP-training, 
including optimization by lambdas to enforce model simplicity (low ranks of TT-cores).

```map_training.ipynb``` contains implementation of proposed SVGD + MAP training algorithm. 

```Generate dataset.ipynb``` contains dataset generation for synthetic experiment with a priori low-rank structure.


### Parameters
Here we present an example set of model paremeters we used to train 2-tt-layers net on common MNIST dataset.

```yaml

model_config:
  resize_shape: (32, 32)                            # resize to obtain good factorization of input dimensions
  in_factors: (4, 4, 4, 4, 4)                       # factorization of input
  l1_ranks: (8, 64, 64, 8)                          # ranks of tt-cores of the 1st layer, presented by maximum rank for particular factorization
  hidd_out_factors: (2, 2, 2, 2, 2)                 # factorization of output of the 1st layer 
  ein_string1: "nabcde,aoiv,bijw,cjkx,dkly,elpz"    # multiplication order for multidimensional tensors
    
  hidd_in_factors: (4, 8)                           # factorization of input of the 2nd layer
  l2_ranks: (16,)                                   # ranks of tt-cores of the second layer
  out_factors: (5, 2)                               # output factorization
  ein_string2: 'nab,aoix,bipy'                      # multiplication order for multidimensional tensors

  a_l: 1                                            # Gamma distribution paremeters for lambdas prior
  b_l: 5
```
