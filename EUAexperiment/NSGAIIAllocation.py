import random
import sys
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
pM = 0.05
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
        # 拥挤度
        self.crowd = 0

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

    def is_control(self, target_chrom):
        if self.get_user_allo() > target_chrom.get_user_allo() and self.get_server_allo() < target_chrom.get_server_allo():
            return True
        return False

    def get_user_allo(self):
        allocated_user_num = user_num - self.user_allocate_list.count(-1)
        return allocated_user_num / user_num

    def get_server_allo(self):
        used_server_num = server_num - self.server_allocate_num.count(0)
        return used_server_num / server_num

    def get_crowd(self, left_chrom, right_chrom):
        user_diff = abs(left_chrom.get_user_allo() - right_chrom.get_user_allo())
        server_diff = abs(left_chrom.get_server_allo() - right_chrom.get_server_allo())
        return user_diff ** 2 + server_diff ** 2 + self.fitness


class NSGAIIAllocate:
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
        # 非支配排序结果
        self.non_dominated_list = []

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
        new_genes = genes[:]
        user_id = random.randint(0, user_num - 1)
        within_servers = users[user_id].within_servers
        server_id = random.choice(within_servers)
        new_genes[user_id] = server_id
        return new_genes

    @staticmethod
    def select(non_dominated_list):
        # 在非支配排序结果中取前chromosome_num个个体
        chrom_list = []
        i = 0
        for li in non_dominated_list:
            for chrom in li:
                chrom_list.append(chrom)
                i += 1
                if i >= chromosome_num:
                    return chrom_list
        return chrom_list

    def generate_next_generation(self):
        for i in range(chromosome_num):
            new_c = self.new_child()
            self.chromosome_list.append(new_c)
            # if new_c.fitness > self.best.fitness:
            #     self.best = new_c
        self.non_dominated_list = self.non_dominated_rank(self.chromosome_list)
        self.chromosome_list = self.select(self.non_dominated_list)
        self.best = self.chromosome_list[0]

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
    def non_dominated_rank(chromosome_list):
        res = []
        first = []
        for p in chromosome_list:
            p.children = []
            p.dominated = 0
            for q in chromosome_list:
                if p.is_control(q):  # 如果p支配q，把q添加到children列表中
                    p.children.append(q)
                elif q.is_control(p):  # 如果p被q支配，则把dominated加1
                    p.dominated += 1

            if p.dominated == 0:
                p.rank = 1  # 如果该个体的dominated为0，则该个体为rank第一级
                first.append(p)
        res.append(first)
        i = 0
        while len(res[i]) > 0:
            Q = []
            for p in res[i]:
                for q in p.children:  # 对所有在children集合中的个体进行排序
                    q.dominated -= 1
                    if q.dominated == 0:  # 如果该个体的支配个数为0，则该个体是非支配个体
                        q.rank = i + 2  # 该个体rank级别为当前最高级别加1。此时i初始值为0，所以要加2
                        Q.append(q)
            res.append(Q)
            i += 1
        return NSGAIIAllocate.calc_crowd(res)

    @staticmethod
    def calc_crowd(non_dominated_list):
        for i in range(len(non_dominated_list)):
            for j in range(len(non_dominated_list[i])):
                if j == 0 or j == len(non_dominated_list[i]) - 1:
                    non_dominated_list[i][j].crowd = sys.maxsize
                else:
                    non_dominated_list[i][j].crowd = \
                        non_dominated_list[i][j].get_crowd(non_dominated_list[i][j - 1], non_dominated_list[i][j + 1])
            non_dominated_list[i] = sorted(non_dominated_list[i], key=lambda x: x.crowd, reverse=True)
        return non_dominated_list

    def tournament_selection(self):
        winner = []
        for i in range(chromosome_num):
            group = random.sample(self.chromosome_list, 5)
            group = sorted(group, key=lambda x: x.fitness, reverse=True)
            winner.append(group[0])
        return winner

    def train(self):
        # 记录程序运行时间
        start_time = time.time()

        # 生成初代染色体
        self.chromosome_list = [Chromosome() for _ in range(chromosome_num * 2)]
        self.chromosome_list = self.tournament_selection()
        self.generation_count = 0
        while self.generation_count < gen_num:
            self.user_result.append(self.best.user_allocated_prop)
            self.server_result.append(self.best.server_used_prop)
            self.generate_next_generation()
            self.pCross += self.generation_count / 10000
            self.pMutate += self.generation_count / 2000
            self.generation_count += 1

        fig = plt.figure()
        plt.plot(list(range(gen_num - 1)), self.user_result[:self.generation_count - 1], c='red', linestyle='--')
        plt.plot(list(range(gen_num - 1)), self.server_result[:self.generation_count - 1], c='red')
        plt.title("NSGAII")
        fig.show()

        # 记录程序结束时间
        end_time = time.time()

        # 程序运行时间
        run_time = end_time - start_time

        print('========NSGAIIAllocation========')
        print('分配用户占所有用户的比例：', self.best.user_allocated_prop)
        print('使用服务器占所有服务器的比例：', self.best.server_used_prop)
        print('程序运行时间：', run_time)

        return self.user_result, self.server_result, run_time
