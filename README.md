# Bayesian DistNet
This repository contains code for modeling algorithm runtime prediction using Bayesian neural networks. We follow a similar structure to [DistNet](https://github.com/KEggensperger/DistNet). We utilize [Pytorch](https://github.com/pytorch/pytorch), along with an open source Pytorch [Bayesian layer library](https://github.com/kumar-shridhar/PyTorch-BayesianCNN).

## Data
We use the same data as [DistNet](https://github.com/KEggensperger/DistNet), which can be found on their page. Place the contents in `./data/`.

## Usage
Run `eval_model.py` inside `./src/` to train a model on a specific scenario. You will need to specify the fold to run, the scenario, the configuration section to use, how many samples per instance, the lower bound percentage, and the net type. 
- See the original DistNet work for the scenarios available
- The configuration sections are listed at `./src/config/training_config.ini`, which specify the training/model configuration parameters
- Each problem instance has up to 100 sample runtimes
- The lower bound percentage is an number from [0, 20, 40, 60, 80], which specifies the percentage of total samples which are treated as a lower bound (censored)
- The net type by default is the original DistNet model, but can be set to `bayes_distnet` for the Bayesian variant.

For example, to train the original DistNet model on fold 1 of the lpg-zeno scenario, with a seed of 1, 8 samples per instance, 20% of total data censored, and using the lognormal distribution:
```shell
python eval_model.py --fold 1 --scenario lpg-zeno --config_section lognormal --seed 1 --num_train_samples 8 --lb 20
```
To train the Bayesian variant of the above:
```shell
python eval_model.py --fold 1 --scenario lpg-zeno --config_section bayesian_lognormal --seed 1 --num_train_samples 8 --lb 20 --net_type bayes_distnet
```

## Model and Data Export
The serialized model, and dataframes for both training/testing will be placed in `./export/`.

## Visualizing the Data
See the worksheet `./worksheets/graph-metrics.ipynb`.