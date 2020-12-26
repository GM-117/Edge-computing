import random
import time
import copy
import math
import numpy as np
import opt


class GA:
    USER_NUM = 0  # 用户（基因）个数
    GROUP_SIZE = 50  # 种群个体数
    MAX_GENERATION = 300  # 最大迭代次数
    PROB_MUT = 0.1  # 变异概率
    fit_score = []  # 适应度得分
    group = []  # 种群，每一行代表一个个体，每一列代表一个user(基因)，每个值代表user分到的服务器，-1表示没分到
    server_group = []  # 服务器资源到种群的映射
    best_score_list = []  # 每一代的最佳适应度得分
    best_generation_individual = []  # 最佳个体user
    best_generation_individual_server = []  # 最佳个体server


def ga_allocation(user_list_par, server_list_par):
    GA.USER_NUM = len(user_list_par)
    GA.fit_score = [0] * GA.GROUP_SIZE
    GA.group = [[0 for col in range(len(user_list_par))] for row in range(GA.GROUP_SIZE)]
    GA.server_group = [[0 for col in range(len(user_list_par))] for row in range(GA.GROUP_SIZE)]

    # 对程序运行时间进行记录
    st_tm = time.time()

    # 初始化种群，服务器资源
    init_group(user_list_par, server_list_par)
    # 为种群中每个user分配初始服务器
    init_allocate()
    # 获取适应度得分
    get_score()

    for i in range(GA.MAX_GENERATION):
        old_group = copy.deepcopy(GA.group)
        old_server_group = copy.deepcopy(GA.server_group)

        opt.selection()  # 选择
        opt.crossover()  # 交叉
        opt.mutation()  # 变异

        GA.group = GA.group + old_group
        GA.server_group = GA.server_group + old_server_group
        get_score()
        selected_idx = np.argsort(GA.fit_score)[:GA.GROUP_SIZE]  # 按适应度排序后取出最佳的group_size个个体
        selected_idx = selected_idx[-1:-1 - GA.GROUP_SIZE:-1]
        get_new_generation(selected_idx)

        best_index = np.array(GA.fit_score).argmax()
        GA.best_score_list.append(GA.fit_score[best_index])
        GA.best_generation_individual.append(GA.group[best_index])
        GA.best_generation_individual_server.append(GA.server_group[best_index])

    best_index = np.array(GA.best_score_list).argmax()
    user_allo_prop, server_used_prop = get_param_by_index(GA.best_generation_individual[best_index],
                                                          GA.best_generation_individual_server[best_index])
    ed_tm = time.time()
    print(GA.fit_score)
    # 程序运行时间
    run_time = ed_tm - st_tm
    print('GaAllocation========================================')
    print('分配用户占所有用户的比例：', user_allo_prop)
    print('使用服务器占所有服务器的比例：', server_used_prop)
    print('程序运行时间：', run_time)

    return user_allo_prop, server_used_prop, run_time


# 适应度得分函数，log(x) & -log(x)
def get_score():
    index = 0
    for user_wait_allocated_list, ser_rem_cap_list in zip(GA.group, GA.server_group):
        # 已分配用户占所有用户的比例
        allocated_users = 0
        for user in user_wait_allocated_list:
            allocated_users += user['is_allocated']
        user_allo_prop = float(allocated_users) / float(len(user_wait_allocated_list))

        # 已使用服务器占所有服务器比例
        used_servers = 0
        for server in ser_rem_cap_list:
            used_servers += server['is_used']
        server_used_prop = float(used_servers) / float(len(ser_rem_cap_list))

        if user_allo_prop == 0 and server_used_prop == 0:
            GA.fit_score[index] = 0
        else:
            user_score = math.log(user_allo_prop, 10)
            server_score = math.log(server_used_prop, 10)
            GA.fit_score[index] = user_score - server_score
        index += 1


# 根据index获取结果参数
def get_param_by_index(user_wait_allocated_list, ser_rem_cap_list):
    # 已分配用户占所有用户的比例
    allocated_users = 0
    for user in user_wait_allocated_list:
        allocated_users += user['is_allocated']
    user_allo_prop = allocated_users / len(user_wait_allocated_list)

    # 已使用服务器占所有服务器比例
    used_servers = 0
    for server in ser_rem_cap_list:
        used_servers += server['is_used']
    server_used_prop = used_servers / len(ser_rem_cap_list)

    return user_allo_prop, server_used_prop


# 初始化种群，每一行都填充为所有user的集合
def init_group(user_list_par, server_list_par):
    for i in range(GA.GROUP_SIZE):
        user_list = copy.deepcopy(user_list_par)
        user_wait_allocated_list = []
        for user in user_list:
            user_info = user.key_info()
            user_info['is_allocated'] = 0
            user_wait_allocated_list.append(user_info)
        GA.group[i] = user_wait_allocated_list

        server_list = copy.deepcopy(server_list_par)
        server_info_list = []
        for server in server_list:
            server_info = server.key_info()
            server_info_list.insert(server_info['id'], {'capacity': server_info['capacity'], 'is_used': 0})
        GA.server_group[i] = server_info_list


# 初始化分配，为种群中每个user分配初始服务器
def init_allocate():
    for user_wait_allocated_list, server_info_list in zip(GA.group, GA.server_group):
        for user in user_wait_allocated_list:
            # 随机选择服务器
            ser_id = random.randint(0, len(server_info_list) - 1)
            ser_rem_cap = server_info_list[ser_id]['capacity']
            user_workload = user['workload']
            # 服务器满足需求时，直接分配
            if (ser_rem_cap[0] >= user_workload[0]) and (ser_rem_cap[1] >= user_workload[1]) and (
                    ser_rem_cap[2] >= user_workload[2]) and (ser_rem_cap[3] >= user_workload[3]):
                if ser_id not in user['within_servers']:
                    continue
                user['is_allocated'] = 1
                user['ser_id'] = ser_id
                server_info_list[ser_id]['is_used'] = 1
                for i in range(4):
                    ser_rem_cap[i] -= user_workload[i]


def get_new_generation(selected_idx):
    new_group = []
    new_server_group = []
    new_fit_score = []
    for index in selected_idx:
        new_group.append(GA.group[index])
        new_server_group.append(GA.server_group[index])
        new_fit_score.append(GA.fit_score[index])
    GA.group = new_group
    GA.server_group = new_server_group
    GA.fit_score = new_fit_score
