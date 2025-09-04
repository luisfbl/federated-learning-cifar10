from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from numpy import ndarray
from typing import List
from collections import OrderedDict
from torch import Tensor
from src.model import CNet, train, test
from src.utils import load_cifar10

class CifarClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, device='cpu'):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items() if val.requires_grad]

    def set_parameters(self, parameters: List[ndarray]):
        state_dict = self.net.state_dict()
        keys = [k for k, v in state_dict.items() if v.requires_grad]
        new_state_dict = OrderedDict({k: Tensor(v) for k, v in zip(keys, parameters)})
        state_dict.update(new_state_dict)
        self.net.load_state_dict(state_dict, strict=True)
        self.net.to(self.device)

    def fit(self, parameters: List[ndarray], config):
        self.set_parameters(parameters)

        epochs = config.get("local_epochs", 1)
        train(self.net, self.trainloader, int(epochs))

        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: List[ndarray], config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader)

        return float(loss), len(self.valloader.dataset), {'accuracy': accuracy}

def client_fn(context: Context) -> Client:
    net = CNet().to('cpu')
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = load_cifar10(partition_id=int(partition_id))

    return CifarClient(net, trainloader, valloader).to_client()

client = ClientApp(client_fn=client_fn)
