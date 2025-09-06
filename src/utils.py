from torch.utils.data import DataLoader
from torch.cuda import is_available
import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset

NUM_CLIENTS = 10
BATCH_SIZE = 32
DEVICE = "cuda" if is_available() else "cpu"

def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    def apply_train_transforms(batch):
        batch["img"] = [train_transforms(img) for img in batch["img"]]
        return batch

    def apply_test_transforms(batch):
        batch["img"] = [test_transforms(img) for img in batch["img"]]
        return batch

    train_data = partition_train_test["train"].with_transform(apply_train_transforms)
    val_data = partition_train_test["test"].with_transform(apply_test_transforms)

    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_data, batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_test_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader
