# ACQL
Official implementation of Automaton Constrained Q-Learning (ACQL)

### Installation

Our code requires Python 3.12+ and additional dependencies. To install these
dependencies with one command, you may use the pixi package management tool
(https://pixi.sh/latest/).

```shell
pixi install
```

### Running ACQL

To run all the experiments included in our paper, you can
run the following scripts.

```shell
python -m acql.scripts.training.train
python -m acql.scripts.training.ablation_train
```

### Hyperparameters

Hyperparameters are summarized for each experiment by dictionaries included with
the definition of each task in the "acql/brax/tasks" subdirectory.

### Acknowledgement

We want to thank the authors of [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL).