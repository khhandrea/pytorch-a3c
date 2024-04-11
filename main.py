from torch import multiprocessing as mp

from network import ActorCriticNetwork
from train import train

STEP_MAX = 5000000
BATCH_SIZE = 128
CPU_NUM = 1 # mp.cpu_count()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    global_network = ActorCriticNetwork()
    global_network.share_memory()
    
    workers = []
    for _ in range(CPU_NUM):
        process = mp.Process(target=train, args=(global_network, STEP_MAX, BATCH_SIZE))
        process.start()
        workers.append(process)

    for idx, worker in enumerate(workers):
        worker.join()