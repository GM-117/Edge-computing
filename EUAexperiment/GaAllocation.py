import random
import time
from Server import Server
from User import User
import matplotlib.pyplot as plt

chromosome_num = 100
user_num = 0
users = []
server_num = 0
servers = []
# 初始交叉概率
pC = 0.5
# 初始变异概率
pM = 0.2
# 迭代轮数
gen_num = 200


def can_allocate(workload, capacity):
    for i in range(len(workload)):
        if capacity[i] < workload[i]:
            return False
    return True


def do_allocate(workload, capacity):
    for i in range(len(workload)):
        capacity[i] -= workload[i]


def copy_int(old_arr: [int]):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr


def copy_server(old_arr: [Server]):
    new_arr = []
    for element in old_arr:
        server = Server(element.latitude, element.longitude, element.id)
        server.coverage = element.coverage
        server.capacity = copy_int(element.capacity)
        new_arr.append(server)
    return new_arr


class Chromosome:
    """
    染色体类
    """

    def __init__(self, genes=None):
        if genes is None:
            genes = []
            for user_id in range(user_num):
                user = users[user_id]
                within_servers = user.within_servers
                server_id = random.choice(within_servers)
                genes.append(server_id)
        self.genes = genes
        self.servers = copy_server(servers)
        self.user_allocated_prop = 0.0
        self.server_used_prop = 1.0
        self.fitness = 0.0
        # 每个用户被分配到的服务器
        self.user_allocate_list = [-1] * user_num
        # 每个服务器分配到的用户数量
        self.server_allocate_num = [0] * server_num
        self.evaluate_fitness()

    def allocate(self, allocated_user_id, allocated_server_id):
        self.user_allocate_list[allocated_user_id] = allocated_server_id
        self.server_allocate_num[allocated_server_id] += 1

    def evaluate_fitness(self):
        # 计算真实分配的用户表和服务器用量
        for i in range(user_num):
            user = users[i]
            server_id = self.genes[i]
            server = self.servers[server_id]
            if can_allocate(user.workload, server.capacity):
                self.allocate(i, server_id)
                do_allocate(user.workload, server.capacity)

        # 已分配用户占所有用户的比例
        allocated_user_num = user_num - self.user_allocate_list.count(-1)
        self.user_allocated_prop = allocated_user_num / user_num

        # 已使用服务器占所有服务器比例
        used_server_num = server_num - self.server_allocate_num.count(0)
        self.server_used_prop = used_server_num / server_num

        self.fitness = self.user_allocated_prop + 1 - self.server_used_prop


class GaAllocate:
    def __init__(self, users_in: [User], servers_in: [Server]):
        self.sumFitness = 0.0
        global users, servers, user_num, server_num
        users = users_in
        servers = servers_in
        user_num = len(users)
        server_num = len(servers)
        self.generation_count = 0
        self.best = Chromosome()
        # 染色体：每个用户想要被分配的 服务器
        self.chromosome_list = []
        self.pCross = pC
        self.pMutate = pM
        # 迭代次数对应的解
        self.user_result = []
        self.server_result = []

    @staticmethod
    def cross(parent1, parent2):
        """
        交叉，把第一个抽出一段基因，放到第二段的相应位置
        :param parent1:
        :param parent2:
        :return:
        """
        index1 = random.randint(0, user_num - 2)
        index2 = random.randint(index1, user_num - 1)
        cross_genes = parent1.genes[index1:index2]
        new_genes = copy_int(parent2.genes)
        new_genes[index1:index2] = cross_genes
        return new_genes

    @staticmethod
    def mutate(genes):
        # 随便改一个
        new_genes = genes[:]
        user_id = random.randint(0, user_num - 1)
        within_servers = users[user_id].within_servers
        server_id = random.choice(within_servers)
        new_genes[user_id] = server_id
        return new_genes

    def generate_next_generation(self):
        # 锦标赛法生成下一代
        for i in range(chromosome_num):
            new_c = self.new_child()
            self.chromosome_list.append(new_c)
            if new_c.fitness > self.best.fitness:
                self.best = new_c
        self.chromosome_list = self.rank(self.chromosome_list)
        self.chromosome_list = self.chromosome_list[0:chromosome_num]

    def new_child(self):
        parent1 = random.choice(self.chromosome_list)
        # 决定是否交叉
        rate = random.random()
        if rate < self.pCross:
            parent2 = random.choice(self.chromosome_list)
            new_genes = self.cross(parent1, parent2)
        else:
            new_genes = copy_int(parent1.genes)
        # 决定是否突变
        rate = random.random()
        if rate < self.pMutate:
            new_genes = self.mutate(new_genes)
        new_chromosome = Chromosome(new_genes)
        return new_chromosome

    @staticmethod
    def rank(chromosome_list):
        new_list = []
        new_list.insert(0, chromosome_list[0])
        for chromosome in chromosome_list:
            for i, ch in enumerate(new_list):
                if chromosome.fitness > ch.fitness:
                    new_list.insert(i, chromosome)
                    break
            new_list.append(chromosome)
        return new_list

    def train(self):
        # 记录程序运行时间
        start_time = time.time()

        # 生成初代染色体
        self.chromosome_list = [Chromosome() for _ in range(chromosome_num)]
        self.generation_count = 0
        while self.generation_count < gen_num:
            self.user_result.append(self.best.user_allocated_prop)
            self.server_result.append(self.best.server_used_prop)
            self.generate_next_generation()
            self.pCross += self.generation_count / 10000
            self.pMutate += self.generation_count / 2000
            self.generation_count += 1

        fig = plt.figure()
        plt.plot(list(range(gen_num-1)), self.user_result[:self.generation_count - 1], c='blue', linestyle='--')
        plt.plot(list(range(gen_num-1)), self.server_result[:self.generation_count - 1], c='blue')
        plt.title("GA")
        fig.show()

        # 记录程序结束时间
        end_time = time.time()

        # 程序运行时间
        run_time = end_time - start_time

        print('========GaAllocation========')
        print('分配用户占所有用户的比例：', self.best.user_allocated_prop)
        print('使用服务器占所有服务器的比例：', self.best.server_used_prop)
        print('程序运行时间：', run_time)

        return self.user_result, self.server_result, run_time
