from achql.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper, AutomatonGoalConditionedWrapper
from achql.brax.envs.wrappers.automaton_transition_rewards_wrapper import AutomatonTransitionRewardsWrapper
from achql.brax.envs.wrappers.automaton_reward_machine_wrapper import AutomatonRewardMachineWrapper
from achql.brax.envs.wrappers.automaton_reward_wrapper import AutomatonRewardWrapper
from achql.brax.envs.wrappers.automaton_cost_wrapper import AutomatonCostWrapper


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
        strip_goal_obs = True,
    )

    return mdp


def make_reward_machine_mdp(task, reward_shaping=False):
    mdp = AutomatonWrapper(
        task.env,
        task.lo_spec,
        task.obs_var,
        strip_goal_obs = True,
    )

    mdp = AutomatonRewardMachineWrapper(
        mdp,
        task.rm_config,
        reward_shaping=reward_shaping
    )

    return mdp


def make_aut_goal_mdp(task, randomize_goals=True):
    mdp = AutomatonGoalConditionedWrapper(
        task.env,
        task.lo_spec,
        task.obs_var,
        randomize_goals=randomize_goals,
    )

    return mdp


def make_aut_goal_cmdp(task, randomize_goals=True, margin=1.0, relu_cost=False, spec_overwriter=lambda x: x):
    mdp = AutomatonGoalConditionedWrapper(
        task.env,
        spec_overwriter(task.lo_spec),
        task.obs_var,
        randomize_goals=randomize_goals,
    )

    cmdp = AutomatonCostWrapper(mdp, margin=margin, relu_cost=relu_cost)

    return cmdp
