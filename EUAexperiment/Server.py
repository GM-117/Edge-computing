import random


class Server:
    # 设置服务器server_id
    server_id = 0

    def __init__(self, latitude, longitude, i):
        self.latitude = latitude
        self.longitude = longitude
        # self.id = Server.server_id
        # Server.server_id = Server.server_id + 1
        self.id = i
        self.coverage = random.randint(450, 750)
        self.capacity = []

    # 服务器数据字典
    def info(self):
        return {'id': self.id, 'latitude': self.latitude, 'longitude': self.longitude, 'coverage': self.coverage,
                'capacity': self.capacity}

    def key_info(self):
        return {'id': self.id, 'capacity': self.capacity}


if __name__ == '__main__':
    server = Server(122, 233)
    print(server.info())
