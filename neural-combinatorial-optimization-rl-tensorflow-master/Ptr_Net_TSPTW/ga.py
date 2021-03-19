import random
import numpy as np
from tqdm import tqdm
from Ptr_Net_TSPTW.config import get_config

config, _ = get_config()

chromosome_num = 50
tasks = []
tasks_num = config.max_length
# 迭代轮数
gen_num = config.gen_num

alpha = config.alpha
beta = config.beta
gama = config.gama

alpha_c = config.alpha_c
alpha_o = config.alpha_o
alpha_b = config.alpha_b
alpha_m = config.alpha_m


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
        self.cpu_sum = 0.0
        self.io_sum = 0.0
        self.bandwidth_sum = 0.0
        self.memory_sum = 0.0
        self.task_priority = 0.0
        self.ns = 0.0
        self.evaluate_fitness()

    def evaluate_fitness(self):
        task_priority_max = 0
        for i in range(tasks_num):
            task_priority_max = max(task_priority_max, tasks[i][4])
        time_used = 0
        ns_ = 0
        cpu_sum = 0
        io_sum = 0
        bandwidth_sum = 0
        memory_sum = 0
        task_priority_sum = 0
        for idx in range(tasks_num):
            i = self.genes[idx]
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

        cpu_sum = cpu_sum / (tasks_num / 4)
        io_sum = io_sum / (tasks_num / 4)
        bandwidth_sum = bandwidth_sum / (tasks_num / 4)
        memory_sum = memory_sum / (tasks_num / 4)
        task_priority_sum = task_priority_sum / (tasks_num / 4)
        ns_prob = ns_ / tasks_num

        reward_1 = 0.25 * (cpu_sum + io_sum + bandwidth_sum + memory_sum)
        reward_2 = task_priority_sum
        reward_3 = ns_prob
        self.fitness = reward_1 + reward_2 + reward_3
        self.cpu_sum = cpu_sum
        self.io_sum = io_sum
        self.bandwidth_sum = bandwidth_sum
        self.memory_sum = memory_sum
        self.task_priority = task_priority_sum
        self.ns = ns_prob


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
        self.cpu_result = []
        self.io_result = []
        self.bandwidth_result = []
        self.memory_result = []
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
            self.cpu_result.append(self.best.cpu_sum)
            self.io_result.append(self.best.io_sum)
            self.bandwidth_result.append(self.best.bandwidth_sum)
            self.memory_result.append(self.best.memory_sum)
            self.task_priority_result.append(self.best.task_priority)
            self.ns_result.append(self.best.ns)
            self.generate_next_generation()
            self.generation_count += 1

        return self.result, self.cpu_result, self.io_result, self.bandwidth_result, self.memory_result, self.task_priority_result, self.ns_result


def do_ga(input_batch):
    result_batch = []
    cpu_result_batch = []
    io_result_batch = []
    bandwidth_result_batch = []
    memory_result_batch = []
    task_priority_result_batch = []
    ns_result_batch = []

    for task in tqdm(input_batch[:32]):
        ga = GaAllocate(task)
        result, cpu_result, io_result, bandwidth_result, memory_result, task_priority_result, ns_result = ga.train()
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

    result = np.mean(result_array, axis=0)
    cpu_result = np.mean(cpu_result_array, axis=0)
    io_result = np.mean(io_result_array, axis=0)
    bandwidth_result = np.mean(bandwidth_result_array, axis=0)
    memory_result = np.mean(memory_result_array, axis=0)
    task_priority_result = np.mean(task_priority_result_array, axis=0)
    ns_result = np.mean(ns_result_array, axis=0)
    return result, cpu_result, io_result, bandwidth_result, memory_result, task_priority_result, ns_result
