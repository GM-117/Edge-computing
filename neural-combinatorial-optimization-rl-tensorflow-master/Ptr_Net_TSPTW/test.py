import random
import numpy as np

tasks = [[1, 2, 3, 4, 5],
         [2, 3, 4, 5, 1],
         [3, 4, 5, 1, 2],
         [4, 5, 1, 2, 3],
         [5, 1, 2, 3, 4]]
task_ = []
for task in tasks:
    task_.append([task[0] + task[1] + task[2] + task[3], task[4]])
print(task_)
