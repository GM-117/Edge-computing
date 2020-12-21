from User import User
from Server import Server
from math import radians, cos, sin, asin, sqrt
import random

# 地球半径
EARTH_REDIUS = 6378.137

class Utils:

    def __init__(self,user_number, server_number, rate):
        self.user_number=user_number
        self.server_number=server_number
        self.rate=rate

    def ger_all_user(self,user_number):
        user_list = []
        file = open("./data/users-melbcbd-generated.csv", 'r+')
        file.readline().strip()  # 数据集的第一行是字段说明信息，不能作为数据，因此跳过
        for i in range(user_number):
            # print(user_number,i)
            line = file.readline().strip()
            latitude, longitude = line.split(',')
            user=User(float(latitude), float(longitude),i)
            user_list.append(user)
        file.close()
        return user_list

    def get_all_server(self,server_number):
        server_list = []
        file = open("./data/site-optus-melbCBD.csv", 'r+')
        file.readline().strip()  # 数据集的第一行是字段说明信息，不能作为数据，因此跳过
        for i in range(server_number):
            line = file.readline().strip()
            result = line.split(',')
            server=Server(float(result[1]), float(result[2]),i)
            server_list.append(server)
        file.close()
        return server_list

    # 判断一个用户是否在一个服务器的覆盖范围内，如果是则返回true,否则返回false
    def judge_cov(self,user, server):
        lat1 = user.latitude
        lng1 = user.longitude
        lat2 = server.latitude
        lng2 = server.longitude
        distance = self.geo_distance(lng1, lat1, lng2, lat2)
        cov = server.coverage
        if distance <= cov:
            return True
        else:
            return False

    # 计算两经纬度点之间距离
    def geo_distance(self,lng1, lat1, lng2, lat2):
        lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
        dlon = lng2 - lng1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        dis = 2 * asin(sqrt(a)) * 6371 * 1000
        return dis

    # 获取用户在哪些服务器的覆盖范围的信息
    def get_within_servers(self,user_list, server_list):
        for user in user_list:
            for server in server_list:
                result = self.judge_cov(user, server)
                if result:
                    user.append_server(server.id)

    # 获取服务器的总的剩余capactiy,比例为rate,可以取值为1,1.5,3
    def get_remain_capacity(self,user_list, rate):
        capacity = [0, 0, 0, 0]
        for user in user_list:
            workload = user.workload
            capacity[0] = capacity[0] + workload[0]
            capacity[1] = capacity[1] + workload[1]
            capacity[2] = capacity[2] + workload[2]
            capacity[3] = capacity[3] + workload[3]
        capacity[0] = capacity[0] * rate
        capacity[1] = capacity[1] * rate
        capacity[2] = capacity[2] * rate
        capacity[3] = capacity[3] * rate
        return capacity

    # 为每个服务器分配capacity
    def allocate_capacity(self,server_list, capacity):
        s_size = len(server_list)
        cpu_max = int(capacity[0] * 1.5 / s_size)
        cpu_min = int(capacity[0] / 2 / s_size)
        io_max = int(capacity[1] * 1.5 / s_size)
        io_min = int(capacity[1] / 2 / s_size)
        bandwidth_max = int(capacity[2] * 1.5 / s_size)
        bandwidth_min = int(capacity[2] / 2 / s_size)
        memory_max = int(capacity[3] * 1.5 / s_size)
        memory_min = int(capacity[3] / 2 / s_size)
        for server in server_list:
            server.capacity = [random.randint(cpu_min, cpu_max), random.randint(io_min, io_max),
                               random.randint(bandwidth_min, bandwidth_max), random.randint(memory_min, memory_max)]




    # user_number为用户数量，server_number为服务器数量，rate为用来随机分配服务器资源的服务器总资源数跟用户所需总资源数的比率
    def init_data(self):
        user_list = self.ger_all_user(self.user_number)
        server_list = self.get_all_server(self.server_number)
        self.get_within_servers(user_list, server_list)
        capacity = self.get_remain_capacity(user_list, self.rate)
        self.allocate_capacity(server_list, capacity)

        return user_list, server_list

