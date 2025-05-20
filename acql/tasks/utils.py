from acql.brax import tasks as brax_tasks
from acql.tasks.base import TaskBase


def get_task(env_name: str, task_name: str, **kwargs) -> TaskBase:
    full_task_name = env_name + task_name
    return get_task_by_name(full_task_name, **kwargs)


def get_task_by_name(full_task_name, extra_tasks={}, **kwargs) -> TaskBase:
    if hasattr(brax_tasks, full_task_name):
        TaskClass = getattr(brax_tasks, full_task_name)
    elif full_task_name in extra_tasks:
        TaskClass = extra_tasks[full_task_name]
    else:
        raise Exception(f"{full_task_name} not found")

    return TaskClass(**kwargs)
