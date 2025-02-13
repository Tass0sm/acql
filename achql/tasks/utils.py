from achql.brax import tasks as brax_tasks
from achql.navix import tasks as navix_tasks
from achql.tasks.base import TaskBase


def get_task(env_name: str, task_name: str) -> TaskBase:
    full_task_name = env_name + task_name

    if hasattr(brax_tasks, full_task_name):
        TaskClass = getattr(brax_tasks, full_task_name)
    elif hasattr(navix_tasks, full_task_name):
        TaskClass = getattr(navix_tasks, full_task_name)
    else:
        raise Exception(f"{full_task_name} not found")

    return TaskClass()
