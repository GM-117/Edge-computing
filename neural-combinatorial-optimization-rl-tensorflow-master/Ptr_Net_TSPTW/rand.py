import numpy as np
import time
import random
from Ptr_Net_TSPTW.config import get_config

tasks = []

config, _ = get_config()
tasks_num = config.max_length

alpha = config.alpha
beta = config.beta
gama = config.gama

gen_num = config.nb_epoch


def get_rand_result(tasks_):
    global tasks
    tasks = tasks_
    result_idx_list = []

    cpu = []
    io = []
    bandwidth = []
    memory = []
    task_priority = []
    timeout = []
    time_use = []
    for task in tasks:
        cpu.append(task[0])
        io.append(task[1])
        bandwidth.append(task[2])
        memory.append(task[3])
        task_priority.append(task[4])
        timeout.append(task[5])
        time_use.append(task[6])

    cpu = np.array(cpu)
    io = np.array(io)
    bandwidth = np.array(bandwidth)
    memory = np.array(memory)
    task_priority = np.array(task_priority)
    timeout = np.array(timeout)
    time_use = np.array(time_use)
    for i in range(tasks_num):
        rand_idx = random.randint(0, 6)
        min_idx = -1
        if rand_idx == 0:
            rand_idx = random.randint(0, 3)
            if rand_idx == 0:
                min_idx = np.argmin(cpu)
            if rand_idx == 1:
                min_idx = np.argmin(io)
            if rand_idx == 2:
                min_idx = np.argmin(bandwidth)
            if rand_idx == 3:
                min_idx = np.argmin(memory)
        if rand_idx == 1 or 3 or 4 or 5 or 6:
            min_idx = np.argmin(task_priority)
        if rand_idx == 2:
            rand_idx = random.randint(0, 2)
            if rand_idx == 0:
                min_idx = np.argmin(timeout)
            if rand_idx == 1:
                min_idx = np.argmin(time_use)
            if rand_idx == 2:
                min_idx = np.argmax(time_use)
        cpu[min_idx] = 10000
        io[min_idx] = 10000
        bandwidth[min_idx] = 10000
        memory[min_idx] = 10000
        task_priority[min_idx] = 10000
        timeout[min_idx] = 10000
        result_idx_list.append(min_idx)

    return get_result(result_idx_list)


def get_result(result_idx_list):
    global tasks

    task_priority_max = 0
    for i in range(tasks_num):
        task_priority_max = max(task_priority_max, tasks[i][4])

    task_priority_sum = 0
    for idx in range(tasks_num):
        i = result_idx_list[idx]
        task_priority = tasks[i][4]
        task_priority = (task_priority / task_priority_max) * (1 - idx / tasks_num)
        task_priority_sum += task_priority

    ns_ = 0
    time_use = 0
    server_run_map = []
    server_remain = [1, 1, 1, 1]
    for idx in result_idx_list:
        task = tasks[idx]
        need = task[:4]
        time_out = task[5]
        time_need = task[6]

        if time_use + time_need > time_out:  # 超时
            ns_ += 1
            continue

        while server_remain[0] < need[0] or server_remain[1] < need[1] or \
                server_remain[2] < need[2] or server_remain[3] < need[3]:
            server_run_map = np.array(server_run_map)
            time_use += 1  # 更新时间
            server_run_map[:, -1] -= 1
            server_run_map = server_run_map.tolist()

            while len(server_run_map) > 0:  # 移除已完成的任务
                min_task_idx = np.argmin(server_run_map, axis=0)[-1]
                min_task = server_run_map[min_task_idx]
                min_need = min_task[:4]
                min_time = min_task[-1]
                if min_time > 0:
                    break
                server_remain = np.add(server_remain, min_need)  # 更新剩余容量
                del server_run_map[min_task_idx]  # 移除任务

        server_run_map.append(task)  # 将新任务加入服务器
        server_remain = np.subtract(server_remain, need)  # 更新服务器剩余容量

    max_time_idx = np.argmax(server_run_map, axis=0)[-1]
    max_time = server_run_map[max_time_idx][-1]
    time_use += max_time
    time_use = time_use / tasks_num
    task_priority_sum = task_priority_sum / (tasks_num // 2)

    ns_prob = ns_ / (tasks_num // 2)

    return 0, time_use, task_priority_sum, ns_prob


def do_rand(input_batch):
    result_batch = []
    time_result_batch = []
    task_priority_result_batch = []
    ns_result_batch = []

    for tasks in input_batch:
        time_start = time.time()
        result, time_result, task_priority_result, ns_result = get_rand_result(tasks)
        time_end = time.time()
        print("rand: ", time_end - time_start)
        result_batch.append(result)
        time_result_batch.append(time_result)
        task_priority_result_batch.append(task_priority_result)
        ns_result_batch.append(ns_result)

    result_array = np.array(result_batch)
    time_result_array = np.array(time_result_batch)
    task_priority_result_array = np.array(task_priority_result_batch)
    ns_result_array = np.array(ns_result_batch)

    result = np.mean(result_array)
    time_result = np.mean(time_result_array, axis=0)
    task_priority_result = np.mean(task_priority_result_array)
    ns_result = np.mean(ns_result_array)

    result = [result] * gen_num
    time_result = [time_result] * gen_num
    task_priority_result = [task_priority_result] * gen_num
    ns_result = [ns_result] * gen_num

    return result, time_result, task_priority_result, ns_result
