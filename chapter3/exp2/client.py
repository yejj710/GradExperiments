import numpy as np
import models
import torch
import threading
import socket
import pickle
import time
 

# 客户端类
class Client(object):
    #构造函数
    def __init__(self, conf, model, train_dataset=None, clinet_id=-1):
        self.conf = conf
        
        self.local_model = models.get_model(self.conf["model_name"])
        
        self.client_id = clinet_id
        
        # self.train_dataset = train_dataset
        # # 按ID对数据集集合进行拆分
        # all_range = list(range(len(self.train_dataset)))
        # data_len = int(len(self.train_dataset) / self.conf['clients_num'])
        # train_indices = all_range[id * data_len: (id + 1) * data_len]
        # 生成数据加载器
        # self.train_loader = torch.utils.data.DataLoader(
        #     # 指定父集合
        #     self.train_dataset,
        #     # 每个batch加载多少样本
        #     batch_size=conf["batch_size"],
        #     # 指定子集合
        #     # sampler定义从数据集中提取样本的策略
        #     sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        # )
 
    # 模型本地训练函数
    def local_train(self, model):
        # 客户端获取服务器的模型，然后通过部分本地数据集进行训练
        for name, param in model.state_dict().items():
            # 用服务器下发的全局模型覆盖本地模型
            self.local_model.state_dict()[name].copy_(param.clone())
        # 定义最优化函数器用户本地模型训练
        optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.conf['lr'],
            momentum=self.conf['momentum']
        )
        # 本地训练模型
        # 设置开启模型训练
        self.local_model.train()
        # 开始训练模型
        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                # 如果可以的话加载到gpu
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                # 梯度初始化为0
                optimizer.zero_grad()
                # 训练预测
                output = self.local_model(data)
                # 计算损失函数cross_entropy交叉熵误差
                loss = torch.nn.functional.cross_entropy(output, target)
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer.step()
            print("Epoch %d done." % e)
        # 创建差值字典（结构与模型参数同规格），用于记录差值
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            # 计算训练后与训练前的差值
            diff[name] = (data - model.state_dict()[name])
            print("Client %d local train done" % self.client_id)
        # 客户端返回差值
        return diff
    
    def local_init(self):
        np.random.seed(self.client_id)
        random_array = np.random.randint(low=-100, high=100, size=2200000)


def send_messages(client_id, num_messages):
    # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client_socket.connect(('127.0.0.1', 12345))
    # init model paramters
    np.random.seed(client_id)
    random_array = np.random.randint(low=-100, high=100, size=2200004)
    split_arrays = np.array_split(random_array, 34)

    for i in range(1, num_messages + 1):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('127.0.0.1', 12345))

        message = (client_id, i-1, split_arrays[i-1])
        # print(f"Client {client_id} sent message: {message}")
        message = pickle.dumps(message)
        # print(len(message))
        client_socket.send(message)
        client_socket.close()
        # time.sleep(1)
    print(f"Client {client_id} finish send message")


def start_clients(num_clients):
    threads = []
    for client_id in range(1, num_clients + 1):
        client_thread = threading.Thread(target=send_messages, args=(client_id, 34))
        threads.append(client_thread)
        client_thread.start()

    for thread in threads:
        thread.join()


def send_full_message(client_id):
    # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client_socket.connect(('127.0.0.1', 12345))
    # init model paramters
    np.random.seed(client_id)
    random_array = np.random.randint(low=-100, high=100, size=2200004)
    # split_arrays = np.array_split(random_array, 34)

    # socket.setdefaulttimeout(20)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    client_socket.connect(('127.0.0.1', 12345))

    message = (client_id, random_array)
    # print(f"Client {client_id} sent message: {message}")
    message = pickle.dumps(message)
    # print(len(message))
    client_socket.send(message)
    client_socket.close()
    print(f"Client {client_id} finish send message")    


def start_full_clients(num_clients):
    threads = []
    for client_id in range(1, num_clients + 1):
        client_thread = threading.Thread(target=send_full_message, args=(client_id,))
        threads.append(client_thread)
        client_thread.start()
        time.sleep(0.1)

    for thread in threads:
        thread.join()


if __name__ == "__main__":

    # 服务器配置
    HOST = '127.0.0.1'
    PORT = 12345
    # np.random.seed(1)
    # random_array = np.random.randint(low=-100, high=100, size=2200000)
    # split_arrays = np.array_split(random_array, 34)
    # tmp = pickle.dumps(split_arrays[0])
    # print(len(tmp))
    # 一层模型大约是0.5MB

    # start_clients(num_clients=100)
    start_full_clients(500)




