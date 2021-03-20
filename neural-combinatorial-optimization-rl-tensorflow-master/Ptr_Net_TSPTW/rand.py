import numpy as np
import time
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

    task_priority_max = 0
    for i in range(tasks_num):
        task_priority_max = max(task_priority_max, tasks[i][1])
    time_used = 0
    ns_ = 0
    cpu_sum = 0
    io_sum = 0
    bandwidth_sum = 0
    memory_sum = 0
    task_priority_sum = 0

    for idx in range(tasks_num):
        i = result_idx_list[idx]
        cpu = tasks[i][0] * 4
        io = tasks[i][1] * 4
        bandwidth = tasks[i][2] * 4
        memory = tasks[i][3] * 4
        task_priority = tasks[i][4]
        timeout = tasks[i][5]
        time_use = tasks[i][6]
        cpu = cpu * (1 - idx / tasks_num)
        io = io * (1 - idx / tasks_num)
        bandwidth = bandwidth * (1 - idx / tasks_num)
        memory = memory * (1 - idx / tasks_num)
        task_priority = (task_priority / task_priority_max) * (1 - idx / tasks_num)
        cpu_sum += cpu
        io_sum += io
        bandwidth_sum += bandwidth
        memory_sum += memory
        task_priority_sum += task_priority
        time_used += time_use
        if timeout < time_used:
            ns_ += 1

    cpu = cpu_sum / (tasks_num / 4)
    io = io_sum / (tasks_num / 4)
    bandwidth = bandwidth_sum / (tasks_num / 4)
    memory = memory_sum / (tasks_num / 4)
    task_priority = task_priority_sum / (tasks_num / 4)
    ns = ns_ / tasks_num

    return 0, cpu, io, bandwidth, memory, task_priority, ns


def do_rand(input_batch):
    result_batch = []
    cpu_result_batch = []
    io_result_batch = []
    bandwidth_result_batch = []
    memory_result_batch = []
    task_priority_result_batch = []
    ns_result_batch = []

    for tasks in input_batch:
        time_start = time.time()
        result, cpu_result, io_result, bandwidth_result, memory_result, task_priority_result, ns_result = get_rand_result(
            tasks)
        time_end = time.time()
        print("rand: ", time_end - time_start)
        result_batch.append(result)
        cpu_result_batch.append(cpu_result)
        io_result_batch.append(io_result)
        bandwidth_result_batch.append(bandwidth_result)
        memory_result_batch.append(memory_result)
        task_priority_result_batch.append(task_priority_result)
        ns_result_batch.append(ns_result)

    result_array = np.array(result_batch)
    cpu_result_array = np.array(cpu_result_batch)
    io_result_array = np.array(io_result_batch)
    bandwidth_result_array = np.array(bandwidth_result_batch)
    memory_result_array = np.array(memory_result_batch)
    task_priority_result_array = np.array(task_priority_result_batch)
    ns_result_array = np.array(ns_result_batch)

    result = np.mean(result_array)
    cpu_result = np.mean(cpu_result_array)
    io_result = np.mean(io_result_array)
    bandwidth_result = np.mean(bandwidth_result_array)
    memory_result = np.mean(memory_result_array)
    task_priority_result = np.mean(task_priority_result_array)
    ns_result = np.mean(ns_result_array)

    result = [result] * gen_num
    cpu_result = [cpu_result] * gen_num
    io_result = [io_result] * gen_num
    bandwidth_result = [bandwidth_result] * gen_num
    memory_result = [memory_result] * gen_num
    task_priority_result = [task_priority_result] * gen_num
    ns_result = [ns_result] * gen_num

    return result, cpu_result, io_result, bandwidth_result, memory_result, task_priority_result, ns_result
