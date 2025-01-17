def get_logical_option_run_ids(task_name):

    # Order matters and corresponds to the AP numbers for each task.
    lo_run_ids_dict = {
        "AntMazeTwoSubgoals": ["731ee5b9391249cd98add1b8049cab4c", "90739cab34244620b6fc70bbabc504de"]
    }

    return lo_run_ids_dict[task_name]
