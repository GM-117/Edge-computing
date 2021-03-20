from Ptr_Net_TSPTW.config import get_config
from Ptr_Net_TSPTW.dataset import DataGenerator
from numpy import *
from Ptr_Net_TSPTW.rand import do_rand

config, _ = get_config()
training_set = DataGenerator(config)
input_batch = training_set.train_batch()
rand_result, rand_cpu_result, rand_io_result, rand_bandwidth_result, rand_memory_result, rand_task_priority_result, rand_ns_result = do_rand(
    input_batch)
print('rand')
print('目标1.1：CPU', mean(rand_cpu_result[-1]))
print('目标1.2：I/O', mean(rand_io_result[-1]))
print('目标1.3：带宽', mean(rand_bandwidth_result[-1]))
print('目标1.4：内存', mean(rand_memory_result[-1]))
print('目标2：任务优先级', mean(rand_task_priority_result[-1]))
print('目标3：超时率', mean(rand_ns_result[-1]))
