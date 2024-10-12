import pickle
from pathlib import Path

import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from server import get_evaluate_fn, get_on_fit_config

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    # loading the MNIST test data on the server to test global model
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    test_set = MNIST("./sim_2_data", train=False, download=True, transform=tr)
    testloader =  DataLoader(test_set, batch_size=32, num_workers=2)
    
    # Defining the strategy for accumulating client models
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0,                                   # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        min_fit_clients=cfg.num_clients_per_round_fit,      # number of clients to sample for fit()
        fraction_evaluate=0.0,                              # similar to fraction_fit, we don't need to use this argument.
        min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
        min_available_clients=cfg.num_clients,                  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),     #retrieves the configuration values 
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),       # tests the global model 
        )
    

    # Clearing the global_model_results
    with open("global_model_results.txt", 'w') as f :
        pass
    
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=cfg.num_rounds), 
                           strategy=strategy)

if __name__ == "__main__":
    main()