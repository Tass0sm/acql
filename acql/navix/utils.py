from achql.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper, AutomatonGoalConditionedWrapper
# from achql.brax.envs.wrappers.automaton_transition_rewards_wrapper import AutomatonTransitionRewardsWrapper
# from achql.brax.envs.wrappers.automaton_reward_machine_wrapper import AutomatonRewardMachineWrapper
# from achql.brax.envs.wrappers.automaton_reward_wrapper import AutomatonRewardWrapper
from achql.navix.envs.wrappers.automaton_wrapper import AutomatonWrapper
from achql.navix.envs.wrappers.automaton_cost_wrapper import AutomatonCostWrapper


def make_cmdp(task, margin=1.0, relu_cost=False):
    automaton_wrapped_env = AutomatonWrapper(
        task.env,
        task.lo_spec,
        task.obs_var,
        augment_obs = False
    )

    cmdp = AutomatonCostWrapper(
        automaton_wrapped_env, margin=margin, relu_cost=relu_cost
    )

    return cmdp
