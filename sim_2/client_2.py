from collections import OrderedDict
from typing import Dict, Tuple
import flwr as fl
from flwr.common import Context
import torch
from flwr.common import NDArrays, Scalar
from model import Net, test, train
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage

class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, trainloader, valloader, num_classes) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = valloader

        # a model that is randomly initialised at first
        self.model = Net(num_classes)

        # figure out if this client has access to GPU support or not
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data
        that belongs to this client. Then, send it back to the server.
        """

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        # a very standard looking optimiser
        optim = torch.optim.SGD(self.model.parameters())

        train(net=self.model,
              trainloader=self.trainloader,
              lr=lr,
              momentum=momentum,
              optimizer=optim,
              epochs=epochs,
              device=self.device)

        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {"accuracy": accuracy}


# Creating new derived class of Dataset because can't apply transformations on non-built in datasets
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    # will be used by random_split() function
    def __len__(self):
        return len(self.dataset)
    
    # method that's called when you try to access an item using indexing
    # will be used by Dataloader during training to get the samples
    def __getitem__(self, idx):
        data, label = self.dataset[idx]  # Get the data and label from the dataset
        if self.transform:                # Check if a transformation is provided
            data = self.transform(data)   # Apply the transformation to the data
        return data, label                # Return the transformed data and the original 

# Extracting subdataset
subset = torch.load('/home/suppra/Desktop/Flower/partitions/ds_2.pt', weights_only=False)

# Applying transformations to the images like typecasting from uint8 to float32, and normalization
tr = Compose([ToPILImage(), ToTensor(), Normalize((0.1307,), (0.3081,))])

# Apply the transformation to your dataset
transformed_dataset = TransformedDataset(subset, transform=tr)

# Splitting into train and validation
trainset, valset = random_split(transformed_dataset, [0.9, 0.1])

# Creating dataloaders
trainloader = DataLoader(trainset, batch_size = 32)
valloader = DataLoader(valset, batch_size = 32)

def client_fn(context: Context):
    # Returns a normal FLowerClient that will use the dataloaders as it's local data.
    return FlowerClient(
        trainloader=trainloader,
        valloader=valloader,
        num_classes=10
    ).to_client()


fl.client.start_client(
    server_address="localhost:8080",
    client_fn=client_fn,
)
