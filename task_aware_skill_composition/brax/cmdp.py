from task_aware_skill_composition.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper, AutomatonGoalConditionedWrapper
from task_aware_skill_composition.brax.envs.wrappers.automaton_transition_rewards_wrapper import AutomatonTransitionRewardsWrapper
from task_aware_skill_composition.brax.envs.wrappers.automaton_cost_wrapper import AutomatonCostWrapper

def make_cmdp(task):
    automaton_wrapped_env = AutomatonWrapper(
        task.env,
        task.lo_spec,
        task.obs_var,
        augment_obs = False
    )

    cmdp = AutomatonCostWrapper(
        automaton_wrapped_env
    )

    return cmdp

def make_aut_goal_cmdp(task):
    mdp = AutomatonGoalConditionedWrapper(
        task.env,
        task.lo_spec,
        task.obs_var,
    )

    cmdp = AutomatonCostWrapper(mdp)

    return cmdp
