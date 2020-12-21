import numpy as np


def ranking(self):
    self.FitV = -self.Y


def selection(self, tourn_size=3):
    aspirants_idx = np.random.randint(self.group_size, size=(self.group_size, tourn_size))
    # 分为group_size组。每组随机取tourn_size个
    aspirants_values = self.FitV[aspirants_idx]
    winner = aspirants_values.argmax(axis=1)  # 按适应度排序，取最高者下标
    sel_index = [aspirants_idx[i, j] for i, j in enumerate(winner)]  # 幸存者下标集合
    self.Chrom = self.Chrom[sel_index, :]  # 幸存者集合
    return self.Chrom


def crossover(self):
    for i in range(0, self.group_size, 2):  # 步长为2
        Chrom1, Chrom2 = self.Chrom[i], self.Chrom[i + 1]
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
    return self.Chrom


def mutation(self):
    for i in range(self.group_size):
        if np.random.rand() < self.prob_mut:
            self.Chrom[i] = reverse(self.Chrom[i])
    return self.Chrom


def reverse(individual):
    n1, n2 = np.random.randint(0, individual.shape[0] - 1, 2)  # 随机选择两个基因位
    if n1 >= n2:
        n1, n2 = n2, n1 + 1
    individual[n1:n2] = individual[n1:n2][::-1]  # 翻转
    return individual
