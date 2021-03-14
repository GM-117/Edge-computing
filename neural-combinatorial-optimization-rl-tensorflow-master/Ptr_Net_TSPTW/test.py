import numpy as np
from Ptr_Net_TSPTW.config import get_config
from Ptr_Net_TSPTW.dataset import DataGenerator

config, _ = get_config()
training_set = DataGenerator(config)
input_instance = training_set.gen_instance()
print(input_instance)
