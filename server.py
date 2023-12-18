#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mer Ott 25 14:59:17 2023

@author: MODAL
"""

# %%
import os
import sys
import time
import flwr as fl
import numpy as np
import SCRIPT_github.PDP2.cifar as cifar
import torch.optim as optim
from typing import List, Tuple
from flwr.common import Metrics
from utils import plot_loss_accuracy_server


strategy = 'FedAvg'  # FedAvg, FedOpt, FedProx, FedYogi

folder_to_save = 'folder'
path_to_save = f'Results/{strategy}/{folder_to_save}'

num_clients = 6
n_round = 30

model = cifar.Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

weights_dict = model.state_dict()
weights = [val.numpy() for _, val in weights_dict.items()]
parameters = fl.common.ndarrays_to_parameters(weights)
parameters_size = sys.getsizeof(parameters)
parameters_size = parameters_size/(1024 ** 2)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


strategy_server = fl.server.strategy.FedAvg(  # FedAvg, FedOpt, FedProx, FedYogi
                    fraction_fit=1.,
                    min_fit_clients=num_clients,
                    min_available_clients=num_clients,
                    initial_parameters=parameters,
                    evaluate_metrics_aggregation_fn=weighted_average)


# %%
if __name__ == "__main__":
    t0 = time.time()
    history = fl.server.start_server(
                config=fl.server.ServerConfig(num_rounds=n_round),
                strategy=strategy_server
                )

    t1 = time.time()
    total_time = t1-t0
    print(f"Total training time: {total_time}")

# %%

plot_loss_accuracy_server(history)
round_accuracy = history.metrics_distributed['accuracy']
accuracy = [round_accuracy[i][1] for i in range(len(round_accuracy))]
system_throughput = 4384 * accuracy[-1] / total_time

os.makedirs(f'{path_to_save}', exist_ok=True)
list_files = os.listdir(f'{path_to_save}')

list_files = [file for file in list_files if file.startswith('Client')]
efficiency_client = []
memory_client = []
for f in list_files:
    with open(f'{path_to_save}/{f}', 'r') as file:
        lines = file.readlines()
        line_eff = [line for line in lines if line.startswith('ENERGY EFFICIENCY')]
        efficiency_client.append(float(line_eff[-1].split(' ')[-2]))
        line_gpu = [line for line in lines if line.startswith('GPU Memory Usage')]
        memory_client.append(float(line_gpu[-1].split(' ')[-2]))

efficiency_client = np.array(efficiency_client)
memory_client = np.array(memory_client)
memory_client *= total_time
memory_client = np.array(memory_client)
EES = accuracy[-1] / np.mean(efficiency_client)
FLE = np.array(accuracy / np.mean(memory_client))
delta_accuracy = [accuracy[0]] + [accuracy[i+1] - accuracy[i] for i in range(len(accuracy)-1)]
comm_efficiency = np.array(delta_accuracy) / parameters_size
comm_efficiency = np.array(comm_efficiency)

os.makedirs(f'{path_to_save}', exist_ok=True)
with open(f'{path_to_save}/Server_performance.txt', 'w') as f:
    f.write(f'Number of clients: {num_clients} \n')
    f.write(f'Number of rounds: {n_round} \n')
    f.write(f'Total training time: {total_time} \n')
    f.write(f'Global model accuracy: {round_accuracy} \n')
    f.write(f'Parameters size: {parameters_size} \n')
    f.write(f'Efficiency Client: {efficiency_client} \n')
    f.write(f'Memory Client: {memory_client} \n')
    f.write(f'Communication efficiency: {comm_efficiency} \n')
    f.write(f'System throughput: {system_throughput} \n')
    f.write(f'Energy Efficiency Server: {EES} \n')
    f.write(f'Federated Learning Efficiency: {FLE} \n')


# %%