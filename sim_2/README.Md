## Simulation 2 : Version 1
One (tiny, baby) step closer to a real-life deployment, this simulation runs the server and clients as different processes on a single node, and removes the need for a VirtualClientEngine. The server runs by default on localhost:8080 and clients connect to the same.

Partitioned data from the MNIST dataset was uploaded as .pt files (can be viewed under /partitions).
Currently, two clients running on different processes on the same node load different partitions.
This is to move away from the simulation methodologies.
The server tests the aggregated global model on the MNIST test dataset.
See /conf/base.yaml for hyperparameters and other variables.

Accuracies over 20 training rounds can be viewed under /global_model_results.txt!