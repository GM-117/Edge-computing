import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt


class AntTsp:
    def __init__(self, city_num,
                 ant_num, generation_num,
                 distance_matrix,
                 alpha=1, beta=2, decay=0.5,
                 ):
        self.city_num = city_num  # 城市数量
        self.ant_num = ant_num  # 蚂蚁数量
        self.generation_num = generation_num  # 迭代次数
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta  # 启发因子的重要程度
        self.decay = decay  # 信息素挥发速度

        self.prob_matrix_distance = 1 / (distance_matrix + 1e-10 * np.eye(city_num, city_num))  # 启发因子，距离的倒数

        self.Phe = np.ones((city_num, city_num))  # 信息素矩阵，每次迭代都会更新
        self.PathTable = np.zeros((ant_num, city_num)).astype(np.int)  # 某一代每个蚂蚁的爬行路径
        self.y = None  # 某一代每个蚂蚁的爬行总距离
        self.generation_best_X, self.generation_best_Y = [], []  # 记录各代的最佳情况
        self.best_x, self.best_y = None, None

    # 计算行走路程
    def cal_total_distance(self, path):
        return sum(
            [distance_matrix[path[i % self.city_num], path[(i + 1) % self.city_num]] for i in range(self.city_num)])

    def run(self):
        for i in range(self.generation_num):  # 对每次迭代
            prob_matrix = (self.Phe ** self.alpha) * (self.prob_matrix_distance ** self.beta)  # 可能性矩阵
            for j in range(self.ant_num):  # 对每个蚂蚁
                self.PathTable[j, 0] = np.random.randint(city_num)  # 随机起点
                for k in range(self.city_num - 1):  # 蚂蚁到达的每个节点
                    ban_set = set(self.PathTable[j, :k + 1])  # 已经经过的点和当前点，不能再次经过
                    allow_list = list(set(range(self.city_num)) - ban_set)  # 在这些点中做选择
                    prob = prob_matrix[self.PathTable[j, k], allow_list]  # 当前点到其他可选点的概率列表
                    prob = prob / prob.sum()  # 概率归一化
                    next_point = np.random.choice(allow_list, size=1, p=prob)[0]  # 在allow_list中以概率prob的方式选取一个元素
                    self.PathTable[j, k + 1] = next_point

            # 计算距离
            y = np.array([self.cal_total_distance(i) for i in self.PathTable])

            # 记录历史最好情况
            index_best = y.argmin()
            x_best, y_best = self.PathTable[index_best, :].copy(), y[index_best].copy()
            self.generation_best_X.append(x_best)
            self.generation_best_Y.append(y_best)

            # 计算需要新增加的信息素
            delta_phe = np.zeros((self.city_num, self.city_num))
            for j in range(self.ant_num):  # 每个蚂蚁
                for k in range(self.city_num - 1):  # 每个节点
                    n1, n2 = self.PathTable[j, k], self.PathTable[j, k + 1]  # 蚂蚁从n1节点爬到n2节点
                    delta_phe[n1, n2] += 1 / y[j]  # 增加的信息素，距离越远，信息素浓度越低
                n1, n2 = self.PathTable[j, self.city_num - 1], self.PathTable[j, 0]  # 蚂蚁从最后一个节点爬回到第一个节点
                delta_phe[n1, n2] += 1 / y[j]  # 增加信息素

            # 信息素衰减+新增信息素
            self.Phe = (1 - self.decay) * self.Phe + delta_phe

        # 取出最好结果
        best_generation = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[best_generation]
        self.best_y = self.generation_best_Y[best_generation]
        return self.best_x, self.best_y


city_num = 40
ant_num = 40
generation_num = 200

points_coordinate = np.random.rand(city_num, 2)  # 城市坐标矩阵
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')  # 城市之间距离矩阵

aca = AntTsp(city_num=city_num, ant_num=ant_num, generation_num=generation_num, distance_matrix=distance_matrix)
best_x, best_y = aca.run()  # 最优路线城市下标集合 、 最优值

# 绘图
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_x, [best_x[0]]])  # 将最优路线首尾相接
best_points_coordinate = points_coordinate[best_points_, :]  # 最优路线对应的坐标
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
pd.DataFrame(aca.generation_best_Y).cummin().plot(ax=ax[1])
plt.show()
