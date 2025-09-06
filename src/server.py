from typing import List, Tuple
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from src.model import Net
from src.utils import DEVICE

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [float(num_examples * m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}

def get_initial_parameters():
    net = Net().to(DEVICE)
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def server_fn(context: Context):
    initial_parameters = ndarrays_to_parameters(get_initial_parameters())

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=10,
        min_evaluate_clients=5,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters
    )
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
