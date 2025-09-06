from collections import OrderedDict
from typing import List

from numpy import ndarray
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from torch import Tensor

from src.model import Net, train, test
from src.utils import load_datasets, DEVICE

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=2)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": accuracy}

def set_parameters(net, parameters: List[ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: Tensor(v) for k, v in params_dict})

    net.load_state_dict(state_dict, strict=False)


def get_parameters(net) -> List[ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def client_fn(context: Context) -> Client:
    net = Net().to(DEVICE)
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = load_datasets(partition_id=int(partition_id))

    return FlowerClient(net, trainloader, valloader).to_client()

app = ClientApp(client_fn=client_fn)
