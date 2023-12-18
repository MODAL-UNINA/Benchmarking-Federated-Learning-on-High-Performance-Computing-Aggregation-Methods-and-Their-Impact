import psutil
import sys
import os
import torch
import GPUtil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection

_interactive_mode = 'ipykernel_launcher' in sys.argv[0] or \
                   (len(sys.argv) == 1 and sys.argv[0] == '')


def is_interactive():
    return _interactive_mode


def myshow(plot_show=True):
    if _interactive_mode and plot_show:
        plt.show()
    else:
        plt.close()


def get_cpu_usage():
    return psutil.cpu_percent()


def get_cpu_times():
    return psutil.cpu_times_percent()


def get_memory_usage():
    return psutil.virtual_memory().percent


def get_swap_usage():
    return psutil.swap_memory().percent


def get_max_memory_allocated(device):
    max_memory_allocated = torch.cuda.max_memory_allocated(device=device)
    max_memory_allocated_MB = max_memory_allocated / (1024 * 1024)

    return max_memory_allocated_MB


def get_metrics(pid):
    cpu_usage = get_cpu_usage()
    cpu_times = get_cpu_times()
    memory_usage = get_memory_usage()
    swap_usage = get_swap_usage()
  
    with open('metrics.txt', 'a') as f:
        f.write(f'cpu_usage: {cpu_usage}\n')
        f.write(f'cpu_times: {cpu_times}\n')
        f.write(f'memory_usage: {memory_usage}\n')
        f.write(f'swap_usage: {swap_usage}\n')
        f.write('\n')


def memory_usage(process=None, device=0):
    if process is None:
        process = psutil.Process(os.getpid())
    print('Process ID:', process.pid)
    memory_info = process.memory_full_info()
    memory_usage_bytes = memory_info.rss 
    vmemory_usage_bytes = memory_info.vms
    processed_data = memory_info.data
    shared_memory = memory_info.shared
    text_memory = memory_info.text
    lib_memory = memory_info.lib
    dirty_memory = memory_info.dirty
    uss = memory_info.uss
    pss = memory_info.pss
    swap = memory_info.swap

    gpus = GPUtil.getGPUs()
    gpus = gpus[device]
    gpus_memory = gpus.memoryUsed
    print(f"GPU memory used: {gpus_memory} MB")

    return memory_usage_bytes / (1024 ** 2), vmemory_usage_bytes / (1024 ** 2), processed_data / (1024 ** 2), shared_memory / (1024 ** 2), text_memory / (1024 ** 2), lib_memory / (1024 ** 2), dirty_memory / (1024 ** 2), uss / (1024 ** 2), pss / (1024 ** 2), swap / (1024 ** 2), gpus_memory


def get_gpu_power_consume(device=0):
    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)

    power_draw = result.stdout.decode('utf-8').strip().split('\n')[device]
    print(f"Consumo energetico: {power_draw} W")
    return power_draw


# %%
def plot_loss_accuracy_server(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    round, measures = zip(*history.losses_distributed)
    plt.plot(round, measures, label="loss distributed")
    plt.xlabel("round")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    round, measures = zip(*history.metrics_distributed['accuracy'])
    plt.plot(round, measures, label="accuracy distributed", color='orange')
    plt.xlabel("round")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
# %%
