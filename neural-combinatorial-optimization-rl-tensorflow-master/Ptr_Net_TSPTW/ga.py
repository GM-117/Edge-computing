import random
import numpy as np
from tqdm import tqdm
from Ptr_Net_TSPTW.config import get_config

config, _ = get_config()

chromosome_num = 50
tasks = []
tasks_num = config.max_length
# 迭代轮数
gen_num = config.nb_epoch

alpha = config.alpha
beta = config.beta
gama = config.gama


def copy_int(old_arr: [int]):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr


class Chromosome:
    """
    染色体类
    """

    def __init__(self, genes=None):
        if genes is None:
            genes = [i for i in range(tasks_num)]
            random.shuffle(genes)
        self.genes = genes
        self.fitness = 0.0
        self.server_ratio = 0.0
        self.task_priority = 0.0
        self.ns = 0.0
        self.evaluate_fitness()

    def evaluate_fitness(self):
        task_priority_max = 0
        for i in range(tasks_num):
            task_priority_max = max(task_priority_max, tasks[i][1])
        time_used = 0
        ns_ = 0
        server_ratio_sum = 0
        task_priority_sum = 0
        for idx in range(tasks_num):
            i = self.genes[idx]
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

        self.fitness = alpha * server_ratio_sum + beta * task_priority_sum + gama * ns_
        self.server_ratio = server_ratio_sum
        self.task_priority = task_priority_sum
        self.ns = ns_


class GaAllocate:
    def __init__(self, input):
        self.sumFitness = 0.0
        global tasks
        tasks = input
        self.generation_count = 0
        self.best = Chromosome()
        # 染色体
        self.chromosome_list = []
        # 迭代次数对应的解
        self.result = []
        self.server_ratio_result = []
        self.task_priority_result = []
        self.ns_result = []

    @staticmethod
    def cross(parent1, parent2):
        """
        交叉，把第一个抽出一段基因，放到第二段的相应位置
        :param parent1:
        :param parent2:
        :return:
        """
        index1 = random.randint(0, tasks_num - 2)
        index2 = random.randint(index1, tasks_num - 1)
        # crossover at the point cxpoint1 to cxpoint2
        pos1_recorder = {value: idx for idx, value in enumerate(parent1.genes)}
        pos2_recorder = {value: idx for idx, value in enumerate(parent2.genes)}
        for j in range(index1, index2):
            value1, value2 = parent1.genes[j], parent2.genes[j]
            pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
            parent1.genes[j], parent1.genes[pos1] = parent1.genes[pos1], parent1.genes[j]
            parent2.genes[j], parent2.genes[pos2] = parent2.genes[pos2], parent2.genes[j]
            pos1_recorder[value1], pos1_recorder[value2] = pos1, j
            pos2_recorder[value1], pos2_recorder[value2] = j, pos2

        return parent1.genes

    @staticmethod
    def mutate(genes):
        # 随便改一个
        index1 = random.randint(0, tasks_num - 2)
        index2 = random.randint(index1, tasks_num - 1)
        genes_left = genes[:index1]
        genes_mutate = genes[index1:index2]
        genes_right = genes[index2:]
        genes_mutate.reverse()
        return genes_left + genes_mutate + genes_right

    def generate_next_generation(self):
        # 下一代
        for i in range(chromosome_num):
            new_c = self.new_child()
            self.chromosome_list.append(new_c)
            if new_c.fitness < self.best.fitness:
                self.best = new_c
        # chaos
        for i in range(chromosome_num // 2):
            chaos = Chromosome()
            if chaos.fitness < self.best.fitness:
                self.best = chaos
            self.chromosome_list.append(chaos)
        # 锦标赛
        self.chromosome_list = self.champion(self.chromosome_list)

    def new_child(self):
        # 交叉
        parent1 = random.choice(self.chromosome_list)
        parent2 = random.choice(self.chromosome_list)
        new_genes = self.cross(parent1, parent2)
        # 突变
        new_genes = self.mutate(new_genes)
        new_chromosome = Chromosome(new_genes)
        return new_chromosome

    @staticmethod
    def champion(chromosome_list):
        group_num = 10  # 小组数
        group_size = 10  # 每小组人数
        group_winner = 5  # 每小组获胜数
        winners = []  # 锦标赛结果
        for i in range(group_num):
            group = []
            for j in range(group_size):
                player = random.choice(chromosome_list)
                player = Chromosome(player.genes)
                group.append(player)
            group = GaAllocate.rank(group)
            winners += group[:group_winner]
        return winners

    @staticmethod
    def rank(chromosome_list):
        for i in range(1, len(chromosome_list)):
            for j in range(0, len(chromosome_list) - i):
                if chromosome_list[j].fitness > chromosome_list[j + 1].fitness:
                    chromosome_list[j], chromosome_list[j + 1] = chromosome_list[j + 1], chromosome_list[j]
        return chromosome_list

    def train(self):
        # 生成初代染色体
        self.chromosome_list = [Chromosome() for _ in range(chromosome_num)]
        self.generation_count = 0
        while self.generation_count < gen_num:
            self.result.append(self.best.fitness)
            self.server_ratio_result.append(self.best.server_ratio)
            self.task_priority_result.append(self.best.task_priority)
            self.ns_result.append(self.best.ns)
            self.generate_next_generation()
            self.generation_count += 1

        return self.result, self.server_ratio_result, self.task_priority_result, self.ns_result


def do_ga(input_batch):
    result_batch = []
    server_ratio_result_batch = []
    task_priority_result_batch = []
    ns_result_batch = []

    for task in tqdm(input_batch):
        ga = GaAllocate(task)
        result, server_ratio_result, task_priority_result, ns_result = ga.train()
        result_batch.append(result)
        server_ratio_result_batch.append(server_ratio_result)
        task_priority_result_batch.append(task_priority_result)
        ns_result_batch.append(ns_result)

    result_array = np.array(result_batch)
    server_ratio_result_array = np.array(server_ratio_result_batch)
    task_priority_result_array = np.array(task_priority_result_batch)
    ns_result_array = np.array(ns_result_batch)

    result = np.mean(result_array, axis=0)
    server_ratio_result = np.mean(server_ratio_result_array, axis=0)
    task_priority_result = np.mean(task_priority_result_array, axis=0)
    ns_result = np.mean(ns_result_array, axis=0)
    return result, server_ratio_result, task_priority_result, ns_result
