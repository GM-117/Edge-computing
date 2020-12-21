import random


class User:
    # 设置最大的负载值
    max_cpu = 10
    max_io = 10
    max_bandwidth = 10
    max_memory = 10
    # 设置用户userId
    userId = 0

    def __init__(self, latitude, longitude,i):
        self.latitude = latitude
        self.longitude = longitude
        self.id=i
        # self.id = User.userId
        # User.userId = User.userId + 1
        self.workload = self.get_random_workload()
        self.within_servers = []

    # 初始化用户的工作负载
    def get_random_workload(self):
        cpu = random.randint(0, User.max_cpu)
        io = random.randint(0, User.max_io)
        bandwidth = random.randint(0, User.max_bandwidth)
        memory = random.randint(0, User.max_memory)
        return (cpu, io, bandwidth, memory)

    # 给用户所在的服务器列表添加服务器元素
    def append_server(self, server_id):
        self.within_servers.append(server_id)

    # 用户数据字典
    def info(self):
        return {'id': self.id, 'latitude': self.latitude, 'longitude': self.longitude, 'workload': self.workload,
                'within_servers': self.within_servers}

    def key_info(self):
        return {'id': self.id, 'workload': self.workload, 'within_servers': self.within_servers}

if __name__ == '__main__':
    user = User(122, 233)
    print(user.info())
