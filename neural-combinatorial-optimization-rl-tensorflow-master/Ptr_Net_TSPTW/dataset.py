import numpy as np
from Ptr_Net_TSPTW.config import get_config


class DataGenerator(object):

    # Initialize a DataGenerator
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.dimension = config.input_dimension
        self.task_num = config.task_num
        self.server_num = config.server_num
        self.server_capacity = config.server_capacity
        self.pretrain = config.pretrain

    # Generate random batch for training procedure
    def train_batch(self):
        tasks_input_batch = []
        servers_input_batch = []
        for _ in range(self.batch_size):
            # Generate random task instance
            tasks, servers = self.gen_instance()
            # Store batch
            tasks_input_batch.append(tasks)
            servers_input_batch.append(servers)
        server_allocate = np.random.randint(low=0, high=self.server_num, size=(self.task_num))
        return tasks_input_batch, servers_input_batch, server_allocate

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
        C = np.random.randint(low=1, high=11, size=(self.task_num, 1))
        O = np.random.randint(low=1, high=11, size=(self.task_num, 1))
        B = np.random.randint(low=1, high=11, size=(self.task_num, 1))
        M = np.random.randint(low=1, high=11, size=(self.task_num, 1))

        Cs = np.average(C) * self.server_capacity
        Os = np.average(O) * self.server_capacity
        Bs = np.average(B) * self.server_capacity
        Ms = np.average(M) * self.server_capacity

        cpu_max = int(Cs * 1.5)
        cpu_min = int(Cs / 2)
        io_max = int(Os * 1.5)
        io_min = int(Os / 2)
        bandwidth_max = int(Bs * 1.5)
        bandwidth_min = int(Bs / 2)
        memory_max = int(Ms * 1.5)
        memory_min = int(Ms / 2)

        Cs = np.random.randint(low=cpu_min, high=cpu_max, size=(self.server_num, 1))
        Os = np.random.randint(low=io_min, high=io_max, size=(self.server_num, 1))
        Bs = np.random.randint(low=bandwidth_min, high=bandwidth_max, size=(self.server_num, 1))
        Ms = np.random.randint(low=memory_min, high=memory_max, size=(self.server_num, 1))

        task_priority = np.random.randint(5, size=(self.task_num, 1))
        time_use = np.random.randint(20, size=(self.task_num, 1))
        time_sum = np.sum(time_use)
        timeout = [[time_sum * (np.random.random_sample() * (1.2 - 0.8) + 0.8) / self.server_capacity * 2]
                   for i in range(self.task_num)]
        timeout = np.array(timeout)
        tasks = np.concatenate((C, O, B, M, task_priority, timeout, time_use), axis=1)
        servers = np.concatenate((Cs, Os, Bs, Ms), axis=1)

        return tasks, servers


if __name__ == "__main__":
    # Config
    config, _ = get_config()
    dataset = DataGenerator(config)

    # Generate some data
    # input_batch = dataset.train_batch()
    input_batch, or_sequence, tw_open, tw_close = dataset.test_batch(seed=0)
    print()
