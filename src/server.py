from typing import List, Tuple
from flwr.common import Context, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}

    accuracies = []
    examples = []

    for num_examples, m in metrics:
        if "accuracy" in m:
            accuracies.append(float(num_examples * m["accuracy"]))
            examples.append(num_examples)

    if not examples:
        return {}

    return {"accuracy": sum(accuracies) / sum(examples)}

strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average
)

def server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=5)
    return ServerAppComponents(strategy=strategy, config=config)

server = ServerApp(server_fn=server_fn)
