from typing import Callable, Iterator
import time
HEADLESS=True

def build_and_execute_tasks(tasks:list[Callable]|Callable,parameters:Iterator,threads_count:int=10,slience=True,waiting_time=0):
    return build_and_execute(parameters,tasks,threads_count,slience,waiting_time)
def build_and_execute(parameters:Iterator,tasks:list[Callable]|Callable,threads_count:int=10,slience=True,waiting_time=0):
    """Initialize multi-threads queue for parallel tasks

    Args:
        parameters (Iterator): Tasks list parameters.

        task (Callable): Task that is going to be call.
                            When create your task, please follow the following format.
                            task(on_checkpoint_reached:Callable[[list],None],arg1,arg2,arg3...)

        threads_count (int, optional): How many site you want to scrape at a time. Defaults to 10.

        slience (bool, optional): Stop printing Tasks Running and Done message. Defaults to True.

        waiting_time (int, optional): Wait number of seconds between each threads start. Defaults to 1.
    """
    def run_task(*args, **kwargs):
        try:
            if isinstance(tasks,list):
                for task in tasks:
                    task(*args, **kwargs)
            else:
                tasks(*args, **kwargs)
        except Exception as e:
            print(e)
    from queue import Queue
    from threading import Thread
    WORKERS_COUNT=threads_count
    queue = Queue()
    def worker():
        while True:
            parameter=queue.get()
            if isinstance(parameter,tuple) or isinstance(parameter,list):
                args=parameter
                kwargs={}
            elif isinstance(parameter,dict):
                args=[]
                kwargs=parameter
            if not slience:
                print(f"Running {args}")
            run_task(*args, **kwargs)
            if not slience:
                print(f"Task done.")
            queue.task_done()
    for i in range(WORKERS_COUNT):
        Thread(target=worker, daemon=True).start()
    def start():
        for row in parameters:
            queue.put(row)
            time.sleep(waiting_time)
        queue.join()
    return start()
