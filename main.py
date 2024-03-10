import multiprocessing as mp

from network import ActorCriticNetwork
from train import train

STEP_MAX = 1000000
BATCH_SIZE = 128
CPU_NUM = mp.cpu_count()

if __name__ == '__main__':
    workers = []
    global_network = ActorCriticNetwork()
    global_network.share_memory()
    
    for _ in range(CPU_NUM):
        process = mp.Process(target=train, args=(global_network, STEP_MAX, BATCH_SIZE))
        process.start()
        workers.append(process)

    for idx, worker in enumerate(workers):
        worker.join()