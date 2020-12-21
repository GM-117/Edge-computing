import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import opt


class GeneticAlgorithmBase:
    def __init__(self, func, city_num,
                 group_size, max_generation, prob_mut):
        self.func = func
        self.group_size = group_size  # 种群数量
        self.max_generation = max_generation  # 最大迭代数
        self.prob_mut = prob_mut  # 变异概率
        self.city_num = city_num  # 城市数量

        self.Chrom = None
        self.X = None  # 每个个体的路线
        self.Y = None  # 每个个体的路线距离
        self.FitV = None  # 个体适应度（距离的负值）

        self.generation_best_X = []
        self.generation_best_Y = []

        self.best_x, self.best_y = None, None


class GaTsp(GeneticAlgorithmBase):
    def __init__(self, func, city_num, group_size, max_generation, prob_mut):
        super().__init__(func, city_num, group_size=group_size, max_generation=max_generation, prob_mut=prob_mut)
        self.chrom_size = self.city_num
        tmp = np.random.rand(self.group_size, self.chrom_size)
        self.Chrom = tmp.argsort(axis=1)  # 初始化种群

    def run(self):
        for i in range(self.max_generation):
            Chrom_old = self.Chrom.copy()
            self.X = self.Chrom
            self.Y = self.func(self.X)
            opt.ranking(self)
            opt.selection(self)  # 选择
            opt.crossover(self)  # 交叉
            opt.mutation(self)  # 变异

            # 将上一代和这一代结合
            self.Chrom = np.concatenate([Chrom_old, self.Chrom], axis=0)
            self.X = self.Chrom
            self.Y = self.func(self.X)
            opt.ranking(self)
            selected_idx = np.argsort(self.Y)[:self.group_size]  # 按适应度排序后取出最佳的group_size个个体
            self.Chrom = self.Chrom[selected_idx, :]

            # 选出这一代的最佳个体
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :].copy())
            self.generation_best_Y.append(self.Y[generation_best_index])

        # 从每一代的最佳个体中选出最佳个体
        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y


city_num = 40  # 城市数量
group_size = 50  # 种群个体数
max_generation = 300  # 最大迭代次数
prob_mut = 0.5  # 变异几率

points_coordinate = np.random.rand(city_num, 2)  # 随机生成城市坐标
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')  # 计算城市之间距离矩阵


def func_transformer(routine):
    return np.array([cal_total_distance(x) for x in routine])


def cal_total_distance(routine):
    return sum([distance_matrix[routine[i % city_num], routine[(i + 1) % city_num]] for i in range(city_num)])


ga_tsp = GaTsp(func_transformer, city_num, group_size, max_generation, prob_mut)
best_points, best_distance = ga_tsp.run()

# 绘图
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
ax[1].plot(ga_tsp.generation_best_Y)
plt.show()
