from flwr.simulation.run_simulation import run_simulation
from src.server import server
from src.client import client

if __name__ == '__main__':
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=10,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    )
