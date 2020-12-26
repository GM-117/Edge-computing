import copy
import numpy as np


class GA:
    group = np.asarray([1, 2, 3, 4, 5])
    fit_score = [3,5,1,6,2]
    GROUP_SIZE = 3
    USER_NUM = 5


def selection():
    aspirants_idx = np.random.randint(GA.USER_NUM, size=(GA.GROUP_SIZE, 3))
    print(aspirants_idx)
    # 分为group_size组。每组随机取tourn_size个
    aspirants_values = np.array(GA.fit_score)[aspirants_idx]
    print(aspirants_values)
    winner = aspirants_values.argmax(axis=1)  # 按适应度排序，取最高者下标
    sel_index = [aspirants_idx[i, j] for i, j in enumerate(winner)]  # 幸存者下标集合
    print(sel_index)

selection()
