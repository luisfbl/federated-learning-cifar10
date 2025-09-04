from flwr_datasets import FederatedDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

NUM_CLIENTS = 5

def load_cifar10(partition_id: int):
    dataset = FederatedDataset(dataset="cifar10", partitioners={'train': NUM_CLIENTS})
    partition = dataset.load_partition(partition_id)

    train_test = partition.train_test_split(test_size=0.2, seed=42)
    torch_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_transform(batch):
        batch['img'] = [torch_transforms(img) for img in batch['img']]
        return batch

    train_test = train_test.with_transform(apply_transform)
    train_loader = DataLoader(dataset=train_test['train'], batch_size=32, shuffle=True)

    valloader = DataLoader(dataset=train_test['test'], batch_size=32)

    testset = dataset.load_split('test').with_transform(apply_transform)
    testloader = DataLoader(testset, batch_size=32)

    return train_loader, valloader, testloader
