# from GAAllocation import ga_allocation

from RandomAllocation import random_allocation
from GreedAllocation import greed_allocation
from GaAllocation import ga_allocation
import random
import matplotlib.pyplot as plt

from Utils import Utils

if __name__ == '__main__':

    server_number = 125
    rate = 1
    "====================根据用户数量画图================================"
    random_user_all_list = []
    greed_user_all_list = []
    ga_user_all_list = []
    user_number_list = [32, 128, 256, 512]

    for user_number in user_number_list:
        print("=======user_number==========", user_number)
        utils = Utils(user_number, server_number, rate)
        user_list, server_list = utils.init_data()
        # for i in user_list:
        #     print(i.key_info())
        # for j in server_list:
        #     print(j.key_info())
        random_user_allo, random_server_used, random_runtime = random_allocation(user_list, server_list)
        greed_user_allo, greed_server_used, greed_runtime = greed_allocation(user_list, server_list)
        ga_user_allo, ga_server_used, ga_runtime = ga_allocation(user_list, server_list)

        random_user_all_list.append(random_user_allo)
        greed_user_all_list.append(greed_user_allo)
        ga_user_all_list.append(ga_user_allo)

    plt.plot(user_number_list, random_user_all_list, 'r-o', user_number_list, greed_user_all_list, 'b-^',
             user_number_list, ga_user_all_list, 'g-s')
    plt.legend(('RANDOM', 'GREED', 'GA'), loc='best')
    plt.xlabel('UserNumber')
    plt.ylabel('AssignUser')
    plt.ylim(bottom=0)
    plt.show()
