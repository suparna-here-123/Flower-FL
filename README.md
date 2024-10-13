## My experiments with the Flower framework for Federated Learning.
In this project, I play around with the "Flower - A Friendly Federated Learning Framework"!
Here is the workflow I've followed to work towards creating a small FL ecosystem on 2 home laptops!

Step 1 : hello_world.ipynb is just me following along with the documentation to simulate an FL ecosystem where clients are spawned on an ad-hoc basis.
Step 2 : sim_1 has "real" non-ephemeral clients running on different processes on the same node as the server, participating in the learning process.
Step 3 : sim_2 has remote clients on different nodes participating in the learning process.

I am still experimenting with different data-partitioning and model averaging strategies!

