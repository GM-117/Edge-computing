import numpy as np
import random
from Ptr_Net_TSPTW.config import get_config

server = []
type = -1
raw_tasks = []
ns_ = 0

config, _ = get_config()
tasks_num = config.task_num

gen_num = config.nb_epoch
tasks = [[] for _ in range(config.server_num)]


# type 0:随机 1:带权贪心 2:资源贪心 3:优先级贪心 4:超时率贪心
def get_rand_result(tasks_, server_):
    global raw_tasks, tasks, server
    raw_tasks, server = tasks_, server_
    tasks = [[] for _ in range(config.server_num)]
    for idx, t in enumerate(tasks_):
        tasks[config.server_allocate[idx]].append(t)
    idx_list = get_idx_list()
    return get_result(idx_list)


def get_idx_list():
    if type == -1:
        return [[]]
    if type == 0:
        return rand_idx_list()
    if type == 1:
        return greed_idx_list()


def rand_idx_list():
    result_idx_list = []
    for task_list in tasks:
        idx_list = list(range(len(task_list)))
        random.shuffle(idx_list)
        result_idx_list.append(idx_list)
    return result_idx_list


def greed_idx_list():
    result_idx_list = []
    for task_list in tasks:
        result_idx_list.append(greed_idx_list_server(task_list))
    return result_idx_list


def greed_idx_list_server(task_list):
    result_idx_list = []
    task = []
    tasks_ = np.array(task_list)
    timout_avg = 2 * np.average(tasks_, axis=0)[-2]
    for task_ in task_list:
        load = (task_[0] + task_[1] + task_[2] + task_[3])
        task.append(load / 40 + task_[4] / 4 + task_[5] / timout_avg)
    task = np.array(task)

    for i in range(len(task_list)):
        min_idx = np.argmin(task)
        task[min_idx] = 10000
        result_idx_list.append(min_idx)

    return result_idx_list


def get_result(result_idx_list):
    global raw_tasks, tasks, server, ns_
    ns_ = 0
    task_priority_max = 0
    for i in range(tasks_num):
        task_priority_max = max(task_priority_max, raw_tasks[i][4])

    task_priority_sum = 0
    result_idx_list_ = []
    for li in result_idx_list:
        result_idx_list_ += li
    for idx in range(tasks_num):
        i = result_idx_list_[idx]
        task_priority = raw_tasks[i][4]
        task_priority = (task_priority / task_priority_max) * (1 - idx / tasks_num)
        task_priority_sum += task_priority

    time_use = []
    for i in range(config.server_num):
        time_use.append(get_result_server(result_idx_list[i], tasks[i], server[i]))
    time_use = np.mean(time_use) / tasks_num

    task_priority_sum = 2 * task_priority_sum / tasks_num
    ns_prob = 2 * ns_ / tasks_num

    result = time_use + task_priority_sum + ns_prob
    return result, time_use, task_priority_sum, ns_prob


def get_result_server(result_idx_list, tasks, server_remain):
    global ns_
    time_use = 0
    server_run_map = []
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

    return time_use


def do_rand(input_batch, servers_input_batch, type_):
    global type
    type = type_
    result_batch = []
    time_result_batch = []
    task_priority_result_batch = []
    ns_result_batch = []

    for tasks, server in zip(input_batch, servers_input_batch):
        result, time_result, task_priority_result, ns_result = get_rand_result(tasks, server)
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
