from flwr.simulation import run_simulation
from src.client import app as client_app
from src.server import app as server_app
from src.utils import NUM_CLIENTS

if __name__ == "__main__":
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_config={"client_resources": {"num_cpus": 3, "num_gpus": 0.0}},
    )
