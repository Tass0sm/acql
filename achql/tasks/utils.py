from achql.brax import tasks as brax_tasks
from achql.navix import tasks as navix_tasks
from achql.tasks.base import TaskBase


def get_task(env_name: str, task_name: str, **kwargs) -> TaskBase:
    full_task_name = env_name + task_name
    return get_task_by_name(full_task_name, **kwargs)


def get_task_by_name(full_task_name, **kwargs) -> TaskBase:
    if hasattr(brax_tasks, full_task_name):
        TaskClass = getattr(brax_tasks, full_task_name)
    elif hasattr(navix_tasks, full_task_name):
        TaskClass = getattr(navix_tasks, full_task_name)
    else:
        raise Exception(f"{full_task_name} not found")

    return TaskClass(**kwargs)
