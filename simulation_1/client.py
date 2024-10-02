import flwr as fl
import torch
from distribute_loaders import distribute

class FlowerClient(fl.client.NumPyClient) :
    def __init__(self, trainloader, valloader) -> None :
        super.__init__()
    

        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Getting parameters
    def fit(self, server_params, config) :
        
