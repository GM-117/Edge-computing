import numpy as np
import GaAllocation as ga
import random


def selection():
    aspirants_idx = np.random.randint(ga.GA.USER_NUM, size=(ga.GA.GROUP_SIZE, ga.GA.USER_NUM // 10))
    # 分为group_size组。每组随机取tourn_size个
    aspirants_values = ga.GA.fit_score[aspirants_idx]
    winner = aspirants_values.argmax(axis=1)  # 按适应度排序，取最高者下标
    sel_index = [aspirants_idx[i, j] for i, j in enumerate(winner)]  # 幸存者下标集合
    ga.get_new_generation(sel_index)


def crossover():
    for i in range(0, ga.GA.GROUP_SIZE, 2):  # 步长为2
        individual1, individual2 = ga.GA.group[i], ga.GA.group[i + 1]

        cxpoint1, cxpoint2 = np.random.randint(0, self.chrom_size - 1, 2)  # 随机抽取两个基因位置
        if cxpoint1 >= cxpoint2:  # 左指针一定要小于右指针
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1 + 1
        # crossover at the point cxpoint1 to cxpoint2
        pos1_recorder = {value: idx for idx, value in enumerate(Chrom1)}
        pos2_recorder = {value: idx for idx, value in enumerate(Chrom2)}
        for j in range(cxpoint1, cxpoint2):
            value1, value2 = Chrom1[j], Chrom2[j]
            pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
            Chrom1[j], Chrom1[pos1] = Chrom1[pos1], Chrom1[j]
            Chrom2[j], Chrom2[pos2] = Chrom2[pos2], Chrom2[j]
            pos1_recorder[value1], pos1_recorder[value2] = pos1, j
            pos2_recorder[value1], pos2_recorder[value2] = j, pos2

        self.Chrom[i], self.Chrom[i + 1] = Chrom1, Chrom2


def mutation():
    i = -1
    for individual, individual_server in zip(ga.GA.group, ga.GA.server_group):
        i += 1
        if np.random.rand() > ga.GA.PROB_MUT:
            continue

        user_id = i
        ser_id = random.randint(0, len(individual_server) - 1)
        user_workload = individual[user_id]['workload']
        ser_rem_cap = individual_server[ser_id]['capacity']

        # server 剩余资源充足
        if (ser_rem_cap[0] >= user_workload[0]) and (ser_rem_cap[1] >= user_workload[1]) and (
                ser_rem_cap[2] >= user_workload[2]) and (ser_rem_cap[3] >= user_workload[3]):
            if ser_id not in individual[user_id]['within_servers']:
                continue
            old_ser_id = individual[user_id]['ser_id']
            if old_ser_id != -1:  # 用户之前已被分配过一个server
                individual_server[old_ser_id]['is_used'] = 0    # server重新标记为未使用
                for i in range(4):
                    individual_server[old_ser_id]['capacity'][i] += user_workload[i]
            for i in range(4):
                ser_rem_cap[i] -= user_workload[i]
