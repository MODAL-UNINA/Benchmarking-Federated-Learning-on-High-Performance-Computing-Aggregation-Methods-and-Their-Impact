# %%
import os
import time
import torch
import psutil
import pickle
import argparse
import flwr as fl
import numpy as np
import pandas as pd
import network as network
from collections import OrderedDict
from typing import Dict, List, Tuple
from utils import memory_usage, get_gpu_power_consume
from flops_profiler.profiler import get_model_profile
from torch.utils.data import TensorDataset, DataLoader


strategy = 'FedAvg'

folder_to_save = 'folder'
path_to_save = f'Results/{strategy}/{folder_to_save}'

server_IP = 'localhost:8080'
num_clients = 6
n_round = 30

worker_id = 0
device = 0
beta = 'iid'

_parser = argparse.ArgumentParser(
    prog="client",
    description="Run the client."
)

_parser.add_argument('--worker_id', type=int, default=worker_id)
_parser.add_argument('--device', type=int, default=device)
_parser.add_argument('--server_IP', type=str, default=server_IP)
args = _parser.parse_known_args()[0]
worker_id = args.worker_id
server_IP = args.server_IP
CVD = args.device
time_list = []


DEVICE: str = torch.device(f"cuda:{CVD}" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


class Client(fl.client.NumPyClient):
    """Flower client implementing image classification using PyTorch."""

    def __init__(
        self,
        model: cifar.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
            ) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        print("Training local model")
        t0 = time.perf_counter()
        cifar.train(self.model, self.trainloader, epochs=5, device=DEVICE, worker_id=worker_id)
        t1 = time.perf_counter()
        time_list.append(t1-t0)
        print("Training done")
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)    
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

# %%


def main(beta) -> None:
    """Load data, start CifarClient."""

    # Load model and data
    model = network.Net()
    model.to(DEVICE)

    if worker_id == 0 or worker_id == 1:
        with open('./mnist_sampling_client/iid/X_train_sampling.pkl', 'rb') as f:
            X = pickle.load(f)[worker_id]
        with open('./mnist_sampling_client/iid/y_train_sampling.pkl', 'rb') as f:
            y = pickle.load(f)[worker_id]

    elif worker_id == 2:
        with open('./digits_client/iid/X_train.pkl', 'rb') as f:
            X = pickle.load(f)[0]
        with open('./digits_client/iid/y_train.pkl', 'rb') as f:
            y = pickle.load(f)[0]

    elif worker_id == 3:
        with open('./digits_client/iid/X_train.pkl', 'rb') as f:
            X = pickle.load(f)[1]
        with open('./digits_client/iid/y_train.pkl', 'rb') as f:
            y = pickle.load(f)[1]

    elif worker_id == 4:
        with open('./semeion_client/iid/X_train.pkl', 'rb') as f:
            X = pickle.load(f)[0]
        with open('./semeion_client/iid/y_train.pkl', 'rb') as f:
            y = pickle.load(f)[0]

    elif worker_id == 5:
        with open('./semeion_client/iid/X_train.pkl', 'rb') as f:
            X = pickle.load(f)[1]
        with open('./semeion_client/iid/y_train.pkl', 'rb') as f:
            y = pickle.load(f)[1]

    # Create an unique testloader for all clients
    test_mnist = pd.read_pickle('../../Dataset_28x28/mnist_images_28x28_test_sampling.pkl')
    test_digits = pd.read_pickle('../../Dataset_28x28/digits_images_28x28_test.pkl')
    test_semeion = pd.read_pickle('../../Dataset_28x28/semeion_images_28x28_test.pkl')
    test_set = np.concatenate((test_mnist, test_digits, test_semeion), axis=0)

    test_mnist_labels = pd.read_pickle('../../Dataset_28x28/mnist_labels_test_sampling.pkl')
    test_digits_labels = pd.read_pickle('../../Dataset_28x28/digits_labels_test.pkl')
    test_semeion_labels = pd.read_pickle('../../Dataset_28x28/semeion_labels_test.pkl')
    test_set_labels = np.concatenate((test_mnist_labels, test_digits_labels, test_semeion_labels), axis=0)

    # Create data loader
    batch_size = 32
    num_examples = {}
    num_examples['trainset'] = len(X)
    num_examples['testset'] = len(test_set)

    trainset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    testset = TensorDataset(torch.from_numpy(test_set).float(), torch.from_numpy(test_set_labels).float())

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    # Start client
    client = Client(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address=server_IP, client=client)
    with torch.cuda.device(CVD):
        flops, macs, params = get_model_profile(model=model, input_shape=(batch_size, 1, 28, 28), args=None, kwargs=None,
                                        print_profile=True, detailed=True, module_depth=-1, top_modules=1, warm_up=10,
                                        as_string=True, output_file=None, ignore_modules=None, func_name='forward')
        print(f'DEVICE: {DEVICE}')
        print(f"PARAMS: {params}")
        print(f"flops: {flops}")
        print(f"MACS: {macs}")

        mean_time = np.mean(time_list[1:])
        print(f'client {worker_id} time : ', mean_time)

    return flops, macs, params, num_examples, mean_time


# %%
if __name__ == "__main__":

    pid_client = os.getpid()
    process = psutil.Process(os.getpid())
    print(process.pid, pid_client)
    t1 = time.perf_counter()
    flops, macs, params, num_examples, mean_time = main(beta)

    flops = float(flops[:-2])
    macs = float(macs[:-5])
    params = float(params[:-1])
    t2 = time.perf_counter()
    execution_time = t2 - t1
    print("time: ", execution_time)

    rss, vms, data, shared, text, lib, dirty, uss, pss, swap, gpu_memory = memory_usage(process, device=CVD)
    power_draw = get_gpu_power_consume(device=CVD)
    power_draw = float(power_draw)

    os.makedirs(f'{path_to_save}', exist_ok=True)
    with open(f'{path_to_save}/Client{worker_id}_{beta}_performance.txt', 'w') as f:
        f.write(f'Server IP: {server_IP.split(":")[0]}\n')
        f.write(f'Port: {server_IP.split(":")[1]}\n')
        f.write(f'Client: {worker_id}\n')
        f.write(f'PID: {pid_client}\n')
        f.write(f'GPU: {CVD}\n')
        f.write(f'params: {params} k\n')
        f.write(f'flops: {flops} M\n')
        f.write(f'MACS: {macs} MMACs\n')
        f.write(f'Execution time: {mean_time} seconds\n')
        f.write(f'Memory Usage RSS: {rss} MB\n')
        f.write(f'Memory Usage VMS: {vms} MB\n')
        f.write(f'Memory Usage DATA: {data} MB\n')
        f.write(f'Memory Usage SHARED: {shared} MB\n')
        f.write(f'Memory Usage TEXT: {text} MB\n')
        f.write(f'Memory Usage LIB: {lib} MB\n')
        f.write(f'Memory Usage DIRTY: {dirty} MB\n')
        f.write(f'Memory Usage USS: {uss} MB\n')
        f.write(f'Memory Usage PSS: {pss} MB\n')
        f.write(f'Memory Usage SWAP: {swap} MB\n')
        f.write(f'GPU Memory Usage: {gpu_memory} MB\n')
        f.write(f'GPU Power Consumption: {power_draw} W\n')
        f.write('\n')
        f.write(f'Number of examples: {num_examples}\n')
        f.write(f'THROUGHPUT: {flops/execution_time} FLOPS\n')
        f.write(f'ENERGY EFFICIENCY: {flops/execution_time/power_draw} FLOPS/W \n')