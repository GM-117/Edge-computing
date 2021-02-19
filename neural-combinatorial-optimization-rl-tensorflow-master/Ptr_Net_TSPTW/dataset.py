from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import os

from Ptr_Net_TSPTW.tsptw_with_ortools import Solver
from Ptr_Net_TSPTW.config import get_config, print_config


class DataGenerator(object):

    # Initialize a DataGenerator
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.dimension = config.input_dimension
        self.max_length = config.max_length
        self.pretrain = config.pretrain

    # Generate random batch for training procedure
    def train_batch(self):
        input_batch = []
        for _ in range(self.batch_size):
            # Generate random task instance
            input_ = self.gen_instance()
            # Store batch
            input_batch.append(input_)
        return input_batch

    # Generate random batch for testing procedure
    def test_batch(self, seed=0):
        # Generate random TSP-TW instance
        input_ = self.gen_instance()
        # Store batch
        input_batch = np.tile(input_, (self.batch_size, 1, 1))
        return input_batch

    # Generate random TSP-TW instance
    def gen_instance(self, seed=0):
        if seed != 0:
            np.random.seed(seed)

        # Randomly generate (max_length) task
        server_ratio = np.random.rand(self.max_length, 1)
        task_priority = np.random.randint(5, size=(self.max_length, 1))
        time_use = np.random.randint(20, size=(self.max_length, 1))
        time_sum = np.sum(time_use)
        timeout = [[time_sum * (np.random.random_sample() * (1.2 - 0.8) + 0.8)] for i in range(self.max_length)]
        timeout = np.array(timeout)
        sequence = np.concatenate((server_ratio, task_priority, timeout, time_use), axis=1)

        return sequence


if __name__ == "__main__":
    # Config
    config, _ = get_config()
    dataset = DataGenerator(config)

    # Generate some data
    # input_batch = dataset.train_batch()
    input_batch, or_sequence, tw_open, tw_close = dataset.test_batch(seed=0)
    print()

    dataset.load_Dumas()
