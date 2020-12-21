# ! /usr/bin/env python
# coding=utf-8

# Describe :将用户随机分配给服务器
# Allocation function：该用户位于该服务器的覆盖范围内
# Allocation function：该服务器有足够的剩余容量容纳该用户

import random
import time
import copy



def random_allocation(user_list_par, server_list_par):
    # 对程序运行时间进行记录
    st_tm = time.time()

    # 对参数进行拷贝，以防改变源数据字典
    user_list = copy.deepcopy(user_list_par)
    server_list = copy.deepcopy(server_list_par)

    # 本算法的两个关键列表，一个是待分配的用户列表，一个是服务器列表
    # 用户列表的每个元素是一个字典,用户列表的数据结构为:[{'id':*, 'workload':*, 'within_servers':*,'is_allocated':*},......]
    # 服务器列表的索引即为服务器的id，列表的每个元素是对应服务器的剩余容量和是否被使用组成的一个列表，数据结构为[['capacity':*,'is_used':*],.......]
    user_wait_allocated_list = []
    ser_rem_cap_list = []

    for user in user_list:
        user_info = user.key_info()
        user_info['is_allocated'] = 0
        user_wait_allocated_list.append(user_info)
    # test print(user_wait_allocated_list)

    for server in server_list:
        server_info = server.key_info()
        ser_rem_cap_list.insert(server_info['id'], {'capacity': server_info['capacity'], 'is_used': 0})
    # test print(ser_rem_cap_list)

    # for循环，为每个用户分配服务器
    for user in user_wait_allocated_list:

        # 从待分配用户列表中获得用户可选的服务器列表
        within_servers = user['within_servers']
        if len(within_servers) > 0:
            # 从待分配用户列表中获得用户需要的负载
            user_workload = user['workload']

            # 从可选服务器中随机选择一个服务器
            index = random.randint(0, len(within_servers) - 1)  #包括左右端
            ser_id = within_servers[index]
            ser_rem_cap = ser_rem_cap_list[ser_id]['capacity']


            while (ser_rem_cap[0] < user_workload[0]) or (ser_rem_cap[1] < user_workload[1]) or (
                    ser_rem_cap[2] < user_workload[2]) or (ser_rem_cap[3] < user_workload[3]):

                # 当服务器无法满足该用户的需求时，将该服务器从该用户的可选服务器中移除,连锁地也会从user_wait_allocated_list中相应地移除
                within_servers.pop(index)

                if len(within_servers) > 0:
                    # 重新选择服务器
                    index = random.randint(0, len(within_servers) - 1)
                    ser_id = within_servers[index]
                    ser_rem_cap = ser_rem_cap_list[ser_id]['capacity']
                else:
                    break
            else:
                user['is_allocated'] = 1
                ser_rem_cap_list[ser_id]['is_used'] = 1
                for i in range(4):
                    ser_rem_cap[i] -= user_workload[i]

    ed_tm = time.time()

    # 已分配用户占所有用户的比例
    allocated_users = 0
    for user in user_wait_allocated_list:
        allocated_users += user['is_allocated']
    user_allo_prop = allocated_users / len(user_wait_allocated_list)

    # 已使用服务器占所有服务器比例
    used_servers = 0
    for server in ser_rem_cap_list:
        used_servers += server['is_used']
    server_used_prop = used_servers / len(ser_rem_cap_list)

    # 程序运行时间
    run_time = ed_tm - st_tm

    print('RandomAllocation========================================')
    print('分配用户占所有用户的比例：', user_allo_prop)
    print('使用服务器占所有服务器的比例：', server_used_prop)
    print('程序运行时间：', run_time)

    return user_allo_prop,server_used_prop,run_time


