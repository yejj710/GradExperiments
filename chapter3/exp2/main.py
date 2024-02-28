import argparse
import json
import random
import threading
# import datasets
from client import *
from server import *
#  reference to https://blog.csdn.net/SAGIRIsagiri/article/details/124048502    用Python实现本地模拟横向联邦学习



if __name__ == '__main__':

    clients = 1
    # threads = []
    start_server('127.0.0.1', 12345, 1)
    server_thread = threading.Thread(target=start_server, args=('127.0.0.1', 12345, clients))
    server_thread.start()
    # threads.append(server_thread)
    client_thread = threading.Thread(target=start_clients, args=(clients))
    client_thread.start()
    # for client_id in range(1, num_clients + 1):
        
    #     threads.append(client_thread)
    #     client_thread.start()

    for thread in [client_thread, server_thread]:
        thread.join()

