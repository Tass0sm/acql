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
the definition of each task in the files "acql/brax/tasks/base.py",
"acql/brax/tasks/simple_maze.py", "acql/brax/tasks/simple_maze_3d.py", and
"acql/brax/tasks/ant_maze.py".

### Acknowledgement

We want to thank the authors of [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL).

### Societal Impact Statement

This work presents a general reinforcement learning algorithm for training
agents in fully observable environments to perform tasks specified using
linear-time temporal logic (LTL). As such, it is primarily foundational research
with limited immediate societal impact, as the current approach assumes
idealized simulation conditions. Nonetheless, this work contributes to the
growing toolbox of formal methods for learning-based control, and could support
the development of reliable robot policies in domains such as warehouse
automation, assistive household robotics, or medical robotics, provided
obstacles to real-world deployment are overcome. The integration of formal task
specifications may help improve trustworthiness, correctness, and
interpretability of robot behavior, which are critical for high-stakes
environments. Potential risks could emerge if such methods are applied in
real-world systems without adequate verification or if the specifications used
are incomplete or misaligned with user intentions. This could result in
unintended or unsafe behaviors, particularly in safety-critical or human-facing
applications. Additionally, if adapted for use in surveillance or military
contexts, the same formal mechanisms could be leveraged to enforce harmful
behaviors. The use of formal specifications may also provide mitigation
opportunities. By enabling precise expression of goals and constraints, these
methods can help surface misalignments earlier in the design process and support
more systematic safety validation. We encourage future work to explore
human-in-the-loop specification, monitoring, and verification strategies to
reduce risks associated with misalignment or misuse.
