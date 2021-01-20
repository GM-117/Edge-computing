from GaAllocation import GaAllocate
from NSGAIIAllocation import NSGAIIAllocate
import matplotlib.pyplot as plt

from Utils import Utils

if __name__ == '__main__':

    user_number = 512
    server_number = 125
    rate = 1
    gen_num = 200

    utils = Utils(user_number, server_number, rate)
    user_list, server_list = utils.init_data()

    nsgaii = NSGAIIAllocate(user_list, server_list)
    nsgaii_user_allo, nsgaii_server_used, nsgaii_runtime = NSGAIIAllocate.train(nsgaii)

    ga = GaAllocate(user_list, server_list)
    ga_user_allo, ga_server_used, ga_runtime = GaAllocate.train(ga)

    fig = plt.figure()
    plt.plot(list(range(gen_num)), nsgaii_user_allo, c='red', linestyle='--')
    plt.plot(list(range(gen_num)), nsgaii_server_used, c='red')
    plt.plot(list(range(gen_num)), ga_user_allo, c='blue', linestyle='--')
    plt.plot(list(range(gen_num)), ga_server_used, c='blue')
    plt.title("NSGAII vs GA")
    fig.show()