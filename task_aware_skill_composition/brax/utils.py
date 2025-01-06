from task_aware_skill_composition.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper, AutomatonGoalConditionedWrapper
from task_aware_skill_composition.brax.envs.wrappers.automaton_transition_rewards_wrapper import AutomatonTransitionRewardsWrapper
from task_aware_skill_composition.brax.envs.wrappers.automaton_reward_machine_wrapper import AutomatonRewardMachineWrapper
from task_aware_skill_composition.brax.envs.wrappers.automaton_reward_wrapper import AutomatonRewardWrapper
from task_aware_skill_composition.brax.envs.wrappers.automaton_cost_wrapper import AutomatonCostWrapper


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


def make_aut_mdp(task):
    mdp = AutomatonWrapper(
        task.env,
        task.lo_spec,
        task.obs_var,
    )

    return mdp


def make_reward_machine_mdp(task):
    mdp = AutomatonWrapper(
        task.env,
        task.lo_spec,
        task.obs_var,
    )

    mdp = AutomatonRewardMachineWrapper(
        mdp,
        task.rm_config,
    )

    return mdp


def make_aut_goal_mdp(task):
    mdp = AutomatonGoalConditionedWrapper(
        task.env,
        task.lo_spec,
        task.obs_var,
    )

    return mdp


def make_aut_goal_cmdp(task):
    mdp = AutomatonGoalConditionedWrapper(
        task.env,
        task.lo_spec,
        task.obs_var,
    )

    cmdp = AutomatonCostWrapper(mdp)

    return cmdp
