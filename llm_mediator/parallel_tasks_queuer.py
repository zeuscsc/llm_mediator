from typing import Callable, Iterator
import time
HEADLESS=True

def execute_queue(iterator:Iterator,threads_count:int=10,slience=True,waiting_time=0):
    def run_task(task:Callable,*args, **kwargs):
        # try:
            task(*args, **kwargs)
        # except Exception as e:
        #     print(e)
    from queue import Queue
    from threading import Thread
    WORKERS_COUNT=threads_count
    queue = Queue()
    def worker():
        while True:
            row=queue.get()
            task=None
            if isinstance(row,tuple) or isinstance(row,list):
                task=row[0]
                parameter=row[1]
            if isinstance(parameter,tuple) or isinstance(parameter,list):
                args=parameter
                kwargs={}
            elif isinstance(parameter,dict):
                args=[]
                kwargs=parameter
            if not slience:
                print(f"Running {args}")
            run_task(task,*args, **kwargs)
            if not slience:
                print(f"Task done.")
            queue.task_done()
    for i in range(WORKERS_COUNT):
        Thread(target=worker, daemon=True).start()
    def start():
        for row in iterator:
            queue.put(row)
            time.sleep(waiting_time)
        queue.join()
    return start()

def build_and_execute_tasks(tasks:list[Callable]|Callable,parameters:Iterator,threads_count:int=10,slience=True,waiting_time=0):
    """Initialize multi-threads queue for parallel tasks
    Args:
        tasks (list[Callable]|Callable): Task that is going to be call.
                            When create your task, please follow the following format.
                            Example:
                                def task(key,value):
                                    return
                                params=[]
                                for key in value:
                                    params.append((key,value))
                                    or
                                    params.append({key=key,value=value})
                                build_and_execute_tasks(task,params)
        parameters (Iterator): Tasks list parameters.
        threads_count (int, optional): How many site you want to scrape at a time. Defaults to 10.
        slience (bool, optional): Stop printing Tasks Running and Done message. Defaults to True.
        waiting_time (int, optional): Wait number of seconds between each threads start. Defaults to 0.
    """
    return build_and_execute(parameters,tasks,threads_count,slience,waiting_time)
def build_and_execute(parameters:Iterator,tasks:list[Callable]|Callable,threads_count:int=10,slience=True,waiting_time=0):
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
