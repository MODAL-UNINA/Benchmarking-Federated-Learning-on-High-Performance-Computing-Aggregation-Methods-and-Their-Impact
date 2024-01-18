# Benchmarking Federated Learning on High Performance Computing Aggregation Methods and Their Impact

This is the implemetation code for the paper (link): Daniela Annunziata, Marzia Canzaniello, Martina Savoia, Salvatore Cuomo, Francesco Piccialli, Benchmarking Federated Learning on High Performance Computing Aggregation Methods and Their Impact.

**Abstract**

Federated Learning (FL) diverges from traditional Machine Learning (ML) models decentralizing data utilization, addressing privacy concerns. 
This approach involves iterative model updates, where individual devices compute gradients based on local data, share updates with a central server, and receive an improved global model. High-Performance Computing (HPC) systems enhance FL efficiency by leveraging parallel processing.

In this study, we aim to explore FL efficiency using four aggregation methods on three datasets across six clients, assess metrics like global model accuracy and communication efficiency, and evaluate FL on HPC. We employ Flower, a versatile FL framework, in our experiments.
Our chosen datasets include MNIST, Digits, and Semeion Handwritten Digit, distributed among two clients each. We utilize NVIDIA GPUs for computation, with aggregation methods such as FedAvg, FedProx, FedOpt, and FedYogi. Metrics include Convergence Time, Global Model Accuracy, Communication Efficiency, and HPC Throughput. The results will provide insights into FL performance, especially in HPC environments, impacting convergence, communication, and resource utilization.

## Acknowledgments

- PNRR project FAIR -  Future AI Research (PE00000013), Spoke 3, under the NRRP MUR program funded by the NextGenerationEU.
- PNRR Centro Nazionale HPC, Big Data e Quantum Computing, (CN\_00000013)(CUP: E63C22000980007), under the NRRP MUR program funded by the NextGenerationEU.
- G.A.N.D.A.L.F. - Gan Approaches for Non-iiD Aiding Learning in Federations, CUP: E53D23008290006, PNRR - Missione 4 “Istruzione e Ricerca” - Componente C2 Investimento 1.1 “Fondo per il Programma Nazionale di Ricerca e Progetti  di Rilevante Interesse Nazionale (PRIN)”.

### Usage: 
To run the Federated Learning process, execute the following command, varying the worker_id number, the device and the server IP.
```python
python client.py --worker_id 1 --device 0 --server_IP "localhost:8080"
```
