# !/bin/env python3

import multiprocessing as mp
import psutil
import pickle
import queue
import time

from utils.logger import logger

SENTINEL = None


def do_work(pending_task, completed_task):
    """ use args and function and run as task while controlling the flow"""
    worker_name = mp.current_process().name
    while True:
        try:
            task = pending_task.get_nowait()
        except queue.Empty:
            time.sleep(0.1)
        else:
            try:
                if task == SENTINEL:
                    # completed_task.put(SENTINEL)
                    break
                # time_start = time.perf_counter()
                work_func = pickle.loads(task["func"])
                result = work_func(**task["task"])
                completed_task.put({work_func.__name__: result})
                # time_end = time.perf_counter() - time_start
                # print(f"{worker_name.upper()} DONE: {round(time_end)} s")
            except Exception as e:
                # print(f"{worker_name.upper()} FAILED: Reason {str(e)}")
                completed_task.put({work_func.__name__: None})
                logger.warning(f"{worker_name.upper()} FAILED: Reason {str(e)}")


def par_proc(job_list, num_cpus=None):
    """ Perform a parallel processing of a list of task"""
    if not num_cpus:
        num_cpus = psutil.cpu_count(logical=False)
    pending_task = mp.Queue()
    completed_task = mp.Queue()
    processes, results = [], []
    # task pointer
    num_tasks = 0
    for job in job_list:
        for task in job["tasks"]:
            exp_jobs = {}
            num_tasks += 1
            exp_jobs.update({'func': pickle.dumps(job['func'])})
            exp_jobs.update({'task': task})
            pending_task.put(exp_jobs)

    num_workers = num_cpus
    for c in range(num_workers):
        pending_task.put(SENTINEL)
    for c in range(num_workers):
        p = mp.Process(target=do_work, args=(pending_task, completed_task))
        p.name = f'worker-{c}'
        processes.append(p)
        p.start()

    completed_task_counter = 0
    while completed_task_counter < num_tasks:
        result = completed_task.get()
        if result == SENTINEL:
            continue
        results.append(result)
        completed_task_counter += 1
    for p in processes:
        p.join(timeout=2)
        p.terminate()
    return results
