# ACHQL
Official implementation of Automaton Constrained Hierarchical Q-Learning (ACHQL) for Safety-Critical Robot Control

### Installation

Our code requires Python 3.12+ and additional dependencies. To install these
dependencies with one command, you may use the pixi package management tool
(https://pixi.sh/latest/).

```shell
pixi install
```

### Running ACHQL

To train an agent with Automaton Constrained Hierarchical Q-Learning, you can
run the `main.py` script. By default, the script instantiates an AntMaze
two-subgoal sequence navigation task.

```shell
python main.py
```

### Acknowledgement

We want to thank the authors of [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL).