from task_aware_skill_composition.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper, AutomatonGoalConditionedWrapper
from task_aware_skill_composition.brax.envs.wrappers.automaton_transition_rewards_wrapper import AutomatonTransitionRewardsWrapper
from task_aware_skill_composition.brax.envs.wrappers.automaton_reward_wrapper import AutomatonRewardWrapper


def make_shaped_reward_mdp(task, multiplier):
    automaton_wrapped_env = AutomatonWrapper(
        task.env,
        task.lo_spec,
        task.obs_var,
        augment_obs = False
    )

    mdp = AutomatonRewardWrapper(
        automaton_wrapped_env,
        multiplier
    )

    return mdp


def make_shaped_reward_mdp2(task, multiplier):
    automaton_wrapped_env = AutomatonWrapper(
        task.env,
        task.lo_spec,
        task.obs_var,
        augment_obs = True,
        augment_with_subgoals = True,
    )

    mdp = AutomatonRewardWrapper(
        automaton_wrapped_env,
        multiplier
    )

    return mdp


def make_aut_goal_mdp(task):
    mdp = AutomatonGoalConditionedWrapper(
        task.env,
        task.lo_spec,
        task.obs_var,
    )

    return mdp
