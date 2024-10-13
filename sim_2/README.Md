## Simulation 2 : Version 1
One (tiny, baby) step closer to a real-life deployment, this simulation runs the server and clients as different processes on a single node, and removes the need for a VirtualClientEngine. The server runs by default on localhost:8080 and clients connect to the same.

Partitioned data from the MNIST dataset was uploaded as .pt files (can be viewed under /partitions).
Currently, two clients running on different processes on the same node load different partitions.
This is to move away from the simulation methodologies.
The server tests the aggregated global model on the MNIST test dataset.
See /conf/base.yaml for hyperparameters and other variables.

## Simulation 2 : Version 2
This real-life deployment has 1 server and 4 clients (2 each on 2 remote nodes).
Data is loaded the same way as version 1.
Keeping all model hyperparameters the same, a ~2% increase in accuracy is observed by increasing the number of clients participating from 2 to just 4!

![Comparison Image](/sim_2/comparisons.png "Comparison between 2 and 4 clients in FL training process")