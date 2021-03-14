import numpy as np
import random
from Ptr_Net_TSPTW.config import get_config

config, _ = get_config()
tasks_num = config.max_length

alpha = config.alpha
beta = config.beta
gama = config.gama

gen_num = config.nb_epoch


def get_rand_result(tasks):
    result_idx_list = []

    server_ratio = []
    task_priority = []
    timeout = []
    for task in tasks:
        server_ratio.append(task[0])
        task_priority.append(task[1])
        timeout.append(task[2])
    server_ratio = np.array(server_ratio)
    task_priority = np.array(task_priority)
    timeout = np.array(timeout)
    for i in range(tasks_num):
        rand_idx = random.randint(0, 2)
        min_idx = -1
        if rand_idx == 0:
            min_idx = np.argmin(server_ratio)
        if rand_idx == 1:
            min_idx = np.argmin(task_priority)
        if rand_idx == 2:
            min_idx = np.argmin(timeout)
        server_ratio[min_idx] = 10000
        task_priority[min_idx] = 10000
        timeout[min_idx] = 10000
        result_idx_list.append(min_idx)

    task_priority_max = 0
    for i in range(tasks_num):
        task_priority_max = max(task_priority_max, tasks[i][1])
    time_used = 0
    ns_ = 0
    server_ratio_sum = 0
    task_priority_sum = 0

    for idx in range(tasks_num):
        i = result_idx_list[idx]
        server_ratio = tasks[i][0]
        task_priority = tasks[i][1]
        timeout = tasks[i][2]
        time_use = tasks[i][3]
        server_ratio = server_ratio * (1 - idx / tasks_num)
        task_priority = (task_priority / task_priority_max) * (1 - idx / tasks_num)
        server_ratio_sum += server_ratio
        task_priority_sum += task_priority
        time_used += time_use
        if timeout < time_used:
            ns_ += 1

    fitness = alpha * server_ratio_sum + beta * task_priority_sum + gama * ns_
    server_ratio = server_ratio_sum
    task_priority = task_priority_sum
    ns = ns_

    return fitness, server_ratio, task_priority, ns


def do_rand(input_batch):
    result_batch = []
    server_ratio_result_batch = []
    task_priority_result_batch = []
    ns_result_batch = []

    for tasks in input_batch:
        result, server_ratio_result, task_priority_result, ns_result = get_rand_result(tasks)
        result_batch.append(result)
        server_ratio_result_batch.append(server_ratio_result)
        task_priority_result_batch.append(task_priority_result)
        ns_result_batch.append(ns_result)

    result_array = np.array(result_batch)
    server_ratio_result_array = np.array(server_ratio_result_batch)
    task_priority_result_array = np.array(task_priority_result_batch)
    ns_result_array = np.array(ns_result_batch)

    result = np.mean(result_array)
    server_ratio_result = np.mean(server_ratio_result_array)
    task_priority_result = np.mean(task_priority_result_array)
    ns_result = np.mean(ns_result_array)

    result = [result] * gen_num
    server_ratio_result = [server_ratio_result] * gen_num
    task_priority_result = [task_priority_result] * gen_num
    ns_result = [ns_result] * gen_num

    return result, server_ratio_result, task_priority_result, ns_result
