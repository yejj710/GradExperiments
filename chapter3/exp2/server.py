import models
import torch
import time
import numpy as np
import threading
import pickle
import socket

GLOBAL_LAYER_BUFFER = {}
mutex_lock = threading.Lock()
ALERADY_AGG = 0
RECV_STATE = None
START_TIME = None


class Server(object):
    def __init__(self, conf, eval_dataset):
        self.conf = conf
        
        self.global_model = models.get_model(self.conf["model_name"])
        # 生成测试集合加载器
        self.eval_loader = torch.utils.data.DataLoader(
          eval_dataset,
          batch_size=self.conf["batch_size"],
          shuffle=True
        )
    
    # weight_accumulator 存储了每个客户端上传参数的变化值
    def model_aggregate(self, weight_accumulator):
        # 遍历服务器的全局模型
        for name, data in self.global_model.state_dict().items():
            # 更新每一次乘以配置文件中的学习率
            update_per_layer = weight_accumulator[name] * self.conf["lambda"]
            # 累加
            if data.type() != update_per_layer.type():
                # 因为update_per_layer的type是floatTensor，所以将其转换为模型的LongTensor（损失精度）
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)
 
    
    def model_eval(self):
        # 开启模型评估模式
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        # 遍历评估数据集合
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            # 获取所有样本总量大小
            dataset_size += data.size()[0]
            # 如果可以的话存储到gpu
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            # 加载到模型中训练
            output = self.global_model(data)
            # 聚合所有损失 cross_entropy 交叉熵函数计算损失
            total_loss += torch.nn.functional.cross_entropy(
              output,
              target,
              reduction='sum'
            ).item()
            # 获取最大的对数概率的索引值，即在所有预测结果中选择可能性最大的作为最终结果
            pred = output.data.max(1)[1]
            # 统计预测结果与真实标签的匹配个数
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        # 计算准确率
        acc = 100.0 * (float(correct) / float(dataset_size))
        # 计算损失值
        total_l = total_loss / dataset_size
 
        return acc, total_l


def _init_layer_buffer(client_nums=1):
    global GLOBAL_LAYER_BUFFER, RECV_STATE
    for i in range(client_nums):
        GLOBAL_LAYER_BUFFER[i+1] = "empty"
    RECV_STATE = [0 for _ in range(34)]
    print(GLOBAL_LAYER_BUFFER)
    print("-"*100)


def aggregation(client_nums=1):
    global GLOBAL_LAYER_BUFFER, ALERADY_AGG, RECV_STATE, START_TIME

    while RECV_STATE[ALERADY_AGG] == client_nums:
        # 可以计算层
        res = np.zeros(64706)
        for k in GLOBAL_LAYER_BUFFER:
            # if not res:
            #     res = GLOBAL_LAYER_BUFFER[k][ALERADY_AGG]
            # else:
            res = np.add(res, GLOBAL_LAYER_BUFFER[k][ALERADY_AGG])
        ALERADY_AGG += 1
        print("finish AGG, ", ALERADY_AGG-1)
        if ALERADY_AGG == 34:
            print("total cost time:", time.time()-START_TIME)
            break


def handle_client(client_socket, client_nums):
    # for _ in range(34):  # Assuming each client sends 5 messages
    global GLOBAL_LAYER_BUFFER, mutex_lock, ALERADY_AGG, RECV_STATE
    client_socket.settimeout(20)
    try:
        received_data = b""
        while len(received_data) < 517814:
            data = client_socket.recv(65500)
            if not data:
                break
            received_data += data

        # print("recv data length: ", len(received_data))
        message = pickle.loads(received_data)
        # print("recv message ,", message)

        client_id, model = message[0], message[2]
        layer_num = message[1]
        mutex_lock.acquire()
        try:
            if GLOBAL_LAYER_BUFFER[client_id] == "empty":
                GLOBAL_LAYER_BUFFER[client_id] = [-1 for _ in range(34)]
                GLOBAL_LAYER_BUFFER[client_id][layer_num] = model
                # ALERADY_AGG += 1
            else:
                GLOBAL_LAYER_BUFFER[client_id][layer_num] = model
            RECV_STATE[layer_num] += 1
            aggregation(client_nums)
        finally:
            # 释放互斥锁
            mutex_lock.release()

        # print(f"Received from Client {client_id}: {data.decode('utf-8')}")
    except socket.timeout:
        print("Timeout: No data received within 20 seconds.")
    # print("*"*100)
    # print(GLOBAL_LAYER_BUFFER)


def start_server(host, port, client_nums):
    global START_TIME
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(10)
    print(f"Server listening on {host}:{port}")

    _init_layer_buffer(client_nums)
    # START_TIME = time.time()
    client_id = 1
    while True:
        client_socket, addr = server_socket.accept()
        # print(f"Accepted connection from {addr}")
        if not START_TIME:
            START_TIME = time.time()
        # 客户端是多线程启动，不一定会遵守顺序
        client_handler = threading.Thread(target=handle_client, args=(client_socket, client_nums))
        # client_handler = threading.Thread(target=handle_client, args=((client_socket),))
        client_handler.start()

        client_id += 1


def full_aggregation(client_nums=1):
    global GLOBAL_LAYER_BUFFER, START_TIME
    res = np.zeros(2200004)
    for k in GLOBAL_LAYER_BUFFER:
        res = np.add(res, GLOBAL_LAYER_BUFFER[k])
    print("total cost time:", time.time()-START_TIME)


def full_handle_client(client_socket, client_nums):
    global GLOBAL_LAYER_BUFFER, mutex_lock, ALERADY_AGG, RECV_STATE
    client_socket.settimeout(20)
    try:
        received_data = b""
        temp = 0
        while len(received_data) < 17600198:
            data = client_socket.recv(65500)
            if not data:
                break
            received_data += data
            # print("recv block {}")
        message = pickle.loads(received_data)
        print("recv message ,", message)
        client_id, model = message[0], message[1]
        # layer_num = message[1]
        mutex_lock.acquire()
        try:
            if GLOBAL_LAYER_BUFFER[client_id] == "empty":
                # GLOBAL_LAYER_BUFFER[client_id] = [-1 for _ in range(34)]
                GLOBAL_LAYER_BUFFER[client_id] = model
                ALERADY_AGG += 1
            else:
                # GLOBAL_LAYER_BUFFER[client_id][layer_num] = model
                raise EOFError("something is wrong")
            # RECV_STATE[layer_num] += 1
            # aggregation(client_nums)
            if ALERADY_AGG == client_nums:
                full_aggregation(client_nums)
        finally:
            # 释放互斥锁
            mutex_lock.release()
    except socket.timeout:
        print("Timeout: No data received within 20 seconds.")


def start_full_server(host, port, client_nums):
    global START_TIME
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(10)
    print(f"Server listening on {host}:{port}")

    _init_layer_buffer(client_nums)
    # START_TIME = time.time()
    client_id = 1
    while True:
        client_socket, addr = server_socket.accept()
        # print(f"Accepted connection from {addr}")
        if not START_TIME:
            START_TIME = time.time()
        # 客户端是多线程启动，不一定会遵守顺序
        client_handler = threading.Thread(target=full_handle_client, args=(client_socket, client_nums))
        # client_handler = threading.Thread(target=handle_client, args=((client_socket),))
        client_handler.start()

        client_id += 1


if __name__ == "__main__":
    # import argparse
    # import json

    # parser = argparse.ArgumentParser(description='Federated Learning')
    # parser.add_argument('-c', '--conf', dest='conf')
    # # 获取所有参数
    # args = parser.parse_args()
    
    # with open(args.conf, 'r', encoding='utf-8') as f:
    #     conf = json.load(f)
    # 1, 5, 10, 50, 100, 500

    # result = [1.0608723163604736, 1.2171602249145508, 2.258134603500366, 4.6717751026153564, 9.567161560058594, 50.33025050163269]
    # start_server('127.0.0.1', 12345, 100)
    start_full_server('127.0.0.1', 12345, 500)
    # [2.4093017578125,  7.436727285385132, 10.224837303161621, 38.355350971221924, 62.35246729850769, 254.4940493106842]


