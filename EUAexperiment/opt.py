import numpy as np
import GaAllocation as ga
import random


def selection():
    aspirants_idx = np.random.randint(ga.GA.GROUP_SIZE, size=(ga.GA.GROUP_SIZE, 5))
    # 分为group_size组。每组随机取tourn_size个
    aspirants_values = np.array(ga.GA.fit_score)[aspirants_idx]
    winner = aspirants_values.argmax(axis=1)  # 按适应度排序，取最高者下标
    sel_index = [aspirants_idx[i, j] for i, j in enumerate(winner)]  # 幸存者下标集合
    ga.get_new_generation(sel_index)


def crossover():
    for i in range(0, ga.GA.GROUP_SIZE, 2):  # 步长为2
        individual_1, individual_2 = ga.GA.group[i], ga.GA.group[i + 1]
        individual_ser_1, individual_ser_2 = ga.GA.server_group[i], ga.GA.server_group[i + 1]
        left, right = np.random.randint(0, ga.GA.USER_NUM - 1, 2)  # 随机抽取两个基因位置
        if left > right:
            left, right = right, left
        for j in range(left, right):
            individual_1[j]['ser_id'], individual_2[j]['ser_id'] = individual_2[j]['ser_id'], individual_1[j]['ser_id']
            ser_id_1, ser_id_2 = individual_1[j]['ser_id'], individual_2[j]['ser_id']
            user_workload = individual_1[j]['workload']
            if ser_id_1 == ser_id_2:  # 父母都没分配服务器或者分配的服务器一致，直接跳过
                continue
            elif ser_id_1 == -1:  # 父没分配服务器，将之前分配的下掉，如果母对应的服务器有资源，给母分配
                ser_rem_cap = individual_ser_1[ser_id_2]['rem_capacity']
                is_used = False
                for idx in range(4):
                    ser_rem_cap[idx] += user_workload[idx]
                    is_used = ser_rem_cap[idx] < individual_ser_1[ser_id_2]['capacity'][idx]
                if not is_used:
                    individual_ser_1[ser_id_2]['is_used'] = 0
                individual_1[j]['is_allocated'] = 0

                ser_rem_cap = individual_ser_2[ser_id_2]['rem_capacity']
                if ser_rem_cap[0] >= user_workload[0] and ser_rem_cap[1] >= user_workload[1] and \
                        ser_rem_cap[2] >= user_workload[2] and ser_rem_cap[3] >= user_workload[3]:
                    for idx in range(4):
                        ser_rem_cap[idx] -= user_workload[idx]
                    individual_ser_2[ser_id_2]['is_used'] = 1
                    individual_2[j]['is_allocated'] = 1

            elif ser_id_2 == -1:
                ser_rem_cap = individual_ser_2[ser_id_1]['rem_capacity']
                is_used = False
                for idx in range(4):
                    ser_rem_cap[idx] += user_workload[idx]
                    is_used = ser_rem_cap[idx] < individual_ser_2[ser_id_1]['capacity'][idx]
                if not is_used:
                    individual_ser_2[ser_id_1]['is_used'] = 0
                individual_2[j]['is_allocated'] = 0

                ser_rem_cap = individual_ser_1[ser_id_1]['rem_capacity']
                if ser_rem_cap[0] >= user_workload[0] and ser_rem_cap[1] >= user_workload[1] and \
                        ser_rem_cap[2] >= user_workload[2] and ser_rem_cap[3] >= user_workload[3]:
                    for idx in range(4):
                        ser_rem_cap[idx] -= user_workload[idx]
                    individual_ser_1[ser_id_1]['is_used'] = 1
                    individual_1[j]['is_allocated'] = 1
            else:  # 父母都分配服务器，如果交换后仍有资源，则分配，否则下掉
                ser_1_rem, ser_2_rem = individual_ser_1[ser_id_1]['rem_capacity'], \
                                       individual_ser_2[ser_id_2]['rem_capacity']
                old_rem_1, old_rem_2 = individual_ser_1[ser_id_2]['rem_capacity'], \
                                       individual_ser_2[ser_id_1]['rem_capacity']
                for idx in range(4):
                    old_rem_1[idx] += user_workload[idx]
                    old_rem_2[idx] += user_workload[idx]
                capacity1 = individual_ser_1[ser_id_2]['capacity']
                capacity2 = individual_ser_2[ser_id_1]['capacity']
                if old_rem_1[0] == capacity1[0] and old_rem_1[1] == capacity1[1] and \
                        old_rem_1[2] == capacity1[2] and old_rem_1[3] == capacity1[3]:
                    individual_ser_1[ser_id_2]['is_used'] = 0
                if old_rem_2[0] == capacity2[0] and old_rem_2[1] == capacity2[1] and \
                        old_rem_2[2] == capacity2[2] and old_rem_2[3] == capacity2[3]:
                    individual_ser_2[ser_id_1]['is_used'] = 0

                if ser_1_rem[0] >= user_workload[0] and ser_1_rem[1] >= user_workload[1] and \
                        ser_1_rem[2] >= user_workload[2] and ser_1_rem[3] >= user_workload[3]:
                    for idx in range(4):
                        ser_1_rem[idx] -= user_workload[idx]
                    individual_ser_1[ser_id_1]['is_used'] = 1
                    individual_1[j]['is_allocated'] = 1
                else:
                    individual_1[j]['ser_id'] = -1
                    individual_1[j]['is_allocated'] = 0

                if ser_2_rem[0] >= user_workload[0] and ser_2_rem[1] >= user_workload[1] and \
                        ser_2_rem[2] >= user_workload[2] and ser_2_rem[3] >= user_workload[3]:
                    for idx in range(4):
                        ser_2_rem[idx] -= user_workload[idx]
                    individual_ser_2[ser_id_2]['is_used'] = 1
                    individual_2[j]['is_allocated'] = 1
                else:
                    individual_2[j]['ser_id'] = -1
                    individual_2[j]['is_allocated'] = 0


def mutation():
    for individual, individual_server in zip(ga.GA.group, ga.GA.server_group):
        for user_id in range(ga.GA.USER_NUM):
            if np.random.rand() > ga.GA.PROB_MUT:
                continue

            user_workload = individual[user_id]['workload']
            within_servers = individual[user_id]['within_servers']
            index = random.randint(0, len(within_servers) - 1)
            ser_id = within_servers[index]
            ser_rem_cap = individual_server[ser_id]['rem_capacity']

            # server 剩余资源充足
            if (ser_rem_cap[0] >= user_workload[0]) and (ser_rem_cap[1] >= user_workload[1]) and (
                    ser_rem_cap[2] >= user_workload[2]) and (ser_rem_cap[3] >= user_workload[3]):
                if ser_id not in individual[user_id]['within_servers']:
                    continue
                old_ser_id = individual[user_id]['ser_id']
                if old_ser_id != -1:  # 用户之前已被分配过一个server
                    is_used = False
                    for j in range(4):
                        individual_server[old_ser_id]['rem_capacity'][j] += user_workload[j]
                        is_used = individual_server[old_ser_id]['rem_capacity'][j] < \
                                  individual_server[old_ser_id]['capacity'][j]
                    if not is_used:
                        individual_server[old_ser_id]['is_used'] = 0  # server重新标记为未使用
                else:
                    individual[user_id]['is_allocated'] = 1
                for j in range(4):
                    ser_rem_cap[j] -= user_workload[j]
                individual[user_id]['ser_id'] = ser_id
                individual_server[ser_id]['is_used'] = 1
