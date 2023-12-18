# %%
import os
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


project = 'Project_name'

output_dir = 'Output_directory'

server_IP = 'server_ip'
num_clients = 6

save_path = f'./{project}/plot_{project}'
os.makedirs(save_path, exist_ok=True)
aggrs = ['FedAvg', 'FedProx', 'FedOpt', 'FedYogi']

strategy = 'FedAvg'

GPUs = '...'  # configuration of GPUs

GPUs_list = [...]  # list of possible GPUs configurations


path = f'./{output_dir}/{strategy}/{GPUs}'

list_file = os.listdir(f'{path}')
server_file = [file for file in list_file if file.startswith('Server')][0]
# %%


def plot_global_accuracy_aggr(aggrs, GPUs, save_path=None):
    accuracies_list = []
    for aggr in aggrs:
        path = f'./{project}/{output_dir}/{aggr}/{GPUs}'

        list_file = os.listdir(f'{path}')
        server_file = [file for file in list_file if file.startswith('Server')][0]
        with open(f'{path}/{server_file}', 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            line_acc = [line for line in lines if line.startswith('Global model accuracy')]
            line_acc = [line.split(':')[-1] for line in line_acc]
            line_acc = line_acc[0].replace(" ", "").replace("\n", "")
            values_list = ast.literal_eval(line_acc)
        rounds = [round_value[0] for round_value in values_list]
        accuracies = [accuracy_value[1] for accuracy_value in values_list]
        accuracies_list.append(accuracies)
    plt.figure(figsize=(10, 6))
    for i, aggr in enumerate(aggrs):
        plt.plot(rounds, accuracies_list[i], '-', label=f'{aggr}')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.xlabel('Rounds', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.legend(fontsize=18)
    if save_path is not None:
        plt.savefig(save_path + f'/global_model_accuracy_aggrs_{GPUs}.png')


def plot_communication_efficiency(path, server_file, numclient, save_path=None):
    with open(f'{path}/{server_file}', 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        line_c_e = [line for line in lines if line.startswith('Communication efficiency')]
        line_c_e = [line.split(':')[-1] for line in line_c_e]
        line_c_e = line_c_e[0].replace("\n", "")
        line_c_e = line_c_e[1:].replace(" ", ",")
        values_list = ast.literal_eval(line_c_e)

    rounds = [i for i in range(len(values_list))]
    communication_efficiency = [value for value in values_list]

    plt.figure(figsize=(10, 6))
    plt.bar(rounds, communication_efficiency, label='Communication efficiency', color='orange')
    plt.xlabel('Rounds', fontsize=18)
    plt.ylabel('Communication efficiency', fontsize=18)
    # plt.title('Communication efficiency', fontsize=18)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path + f'/communication_efficiency_num_clients={num_clients}.png')
    plt.show()


def plot_communication_efficiency_aggr(aggrs, GPUs, save_path=None):
    communication_efficiency_aggrs = []

    for aggr in aggrs:
        path = f'/{output_dir}/{aggr}/{GPUs}'

        list_file = os.listdir(f'{path}')
        server_file = [file for file in list_file if file.startswith('Server')][0]
        with open(f'{path}/{server_file}', 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            line_c_e = [line for line in lines if line.startswith('Communication efficiency')]
            line_c_e = [line.split(':')[-1] for line in line_c_e]
            line_c_e = line_c_e[0].replace("\n", "")
            line_c_e = line_c_e[1:].replace(" ", ",")
            values_list = ast.literal_eval(line_c_e)
        rounds = [i for i in range(len(values_list))]
        communication_efficiency = [value for value in values_list]
        communication_efficiency_aggrs.append(communication_efficiency)
    bar_width = 0.2
    round_spacing = 0.5
    positions = np.arange(len(rounds)) * (len(aggrs) * bar_width + round_spacing)

    plt.figure(figsize=(15, 6))
    for i, aggrr in enumerate(aggrs):
        plt.bar(positions + i * bar_width, communication_efficiency_aggrs[i], width=bar_width, label=f'{aggrr}')
    plt.xlabel('Rounds', fontsize=18)
    plt.yscale('log')
    plt.ylabel('Communication efficiency', fontsize=18)
    x_ticks = [1, 5, 10, 15, 20, 25, 30]
    plt.xticks(x_ticks, x_ticks, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Communication efficiency', fontsize=18)
    plt.legend(fontsize=18)

    plt.xticks(positions + (len(aggrs) * bar_width) / 2, [int(round+1) for round in rounds])

    if save_path is not None:
        plt.savefig(save_path + f'/communication_efficiency_aggrs_num_clients={num_clients}.png')

    plt.show()


# %% -----------------------------------------------------------------------------------------------

plot_global_accuracy_aggr(aggrs, num_clients, GPUs, save_path=save_path)
plot_communication_efficiency(path, server_file, num_clients, save_path=save_path)
plot_communication_efficiency_aggr(aggrs, num_clients, aggrs, GPUs, save_path=save_path)

# %%


def plot_execution_time():
    strategies = ['FedAvg', 'FedProx', 'FedOpt', 'FedYogi']
    execution_dict = {}
    for strategy in strategies:
        path_pdp2 = ['list fo .txt files with information about execution time']
        # one gpu time
        with open(f'{path_pdp2[0]}', 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            line_fle = [line for line in lines if line.startswith('Federated Learning Efficiency')]

        fle = pd.DataFrame(columns=['path', 'fle'])
        for path in path_pdp2:
            with open(f'{path}', 'r') as file:
                lines = file.readlines()
                line_fle = [line for line in lines if line.startswith('Federated Learning Efficiency')]
                fle = pd.concat([fle, pd.DataFrame([[path[83:120], float(line_fle[-1].split(' ')[-2])]], columns=['path', 'fle'])], ignore_index=True)
        execution_dict[f'{strategy}'] = fle

    plt.figure(figsize=(10, 6))
    x = [1, 2, 3, 4, 5, 6]
    for strategy in strategies:
        plt.plot(x, execution_dict[strategy].fle, '-', marker='s', label=f'{strategy}')
        plt.xlabel('Number workers', fontsize=18)
        plt.yscale('log')
        plt.ylabel('Federated Learning Efficiency', fontsize=18)
        plt.legend(fontsize=14)
    return execution_dict


execution_dict = plot_execution_time_pdp2()